#!/usr/bin/env python3
"""
renter_share2_models.py -- the second renter-share regression for
plan/RENTER-SHARE-2.md: more covariates, the post-S2 cluster outcomes,
statistics only (no figures).

UNIT. A body x locate-route cell from output/renter_share_2/outcomes_clusters.csv
(built by analysis/renter_share2_outcomes.py off the post-S2, role_audit
speaker-cluster table). Route is a fixed-effect stratum, never pooled.

OUTCOMES (four binomial outcomes, each with its own denominator):
  pub_nimby / pub_n    public NIMBY clusters out of public clusters with a stance
  pub_yimby / pub_n    public YIMBY clusters, same denominator
  all_nimby / all_n    public + dais NIMBY out of all clusters with a stance
  all_yimby / all_n    public + dais YIMBY, same denominator

MODEL. Bayesian binomial GLM, cell-level overdispersion, route stratum, every
covariate standardised over the fitted cells:

    y_i        ~ Binomial(N_i, p_i)
    logit p_i  = alpha + sum_j beta_j z_ji + gamma_route(i) + u_i
    u_i        ~ Normal(0, sigma_cell)        sigma_cell ~ HalfNormal(0.75)
    alpha      ~ Normal(0, 1.5)               gamma      ~ Normal(0, 1)

Three specifications, because p ~ 8 against n ~ 22 bodies cannot support one
unregularised model with everything in it:

  (a) base      beta ~ Normal(0, 1) on renter_share + log_income
  (b) base+dens beta ~ Normal(0, 1) on renter_share + log_income + log_density
  (c) full-hs   the regularised horseshoe of Piironen & Vehtari (2017) on all
                covariates kept by the VIF screen:
                    beta_j      = z_j * tau * lambda_tilde_j
                    z_j         ~ Normal(0, 1)
                    lambda_j    ~ HalfCauchy(1)
                    tau         ~ HalfCauchy(tau0),
                       tau0     = p0/(p - p0) * sigma / sqrt(n),  p0 = 2,
                                  sigma = 2 (the logistic pseudo-sd)
                    c2          ~ InvGamma(nu/2, nu/2 * s^2), nu = 4, s = 2
                    lambda_tilde_j = sqrt(c2 lambda_j^2 / (c2 + tau^2 lambda_j^2))
                The slab keeps a genuinely large coefficient identified while
                tau0 pulls the rest to zero; this is the "which, if any, of
                these survives regularisation" question stated as a prior.

Reported per fit: posterior median, 90 % and 95 % equal-tailed intervals,
Pr(beta > 0), max R-hat, min ESS, divergences, and PSIS-LOO ELPD, plus an
az.compare ranking of (a)(b)(c) within each outcome. Sanity checks with no
hierarchy and no priors: OLS on the observed share with HC3 standard errors,
and a binomial logistic IRLS with an HC3-type sandwich (both implemented here;
statsmodels is not installed in this venv).

Writes output/renter_share_2/models/{model_results.json, model_table.csv,
loo_table.csv, ols_hc3.csv}. No figures (the brief asked for stats only).
No DB write, no spend, nothing committed.

    nohup .venv/bin/python analysis/renter_share2_models.py \
        > output/renter_share_2/models.log 2>&1 &
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
OUT = ROOT / "output" / "renter_share_2"
MOUT = OUT / "models"
STAMP = ("EXPLORATORY — stance κ ≥ 0.6 hand-coding gate not passed "
         "(plan/SHOWCASE.md 4a); post-S2 cluster unit, role_audit roles")
SEED = 20261019
DRAWS, TUNE, CHAINS = 2000, 1500, 4
TUNE_HS = 4000          # the horseshoe funnel; see fit()
MIN_N = 5
ROUTE_REF = "classifier"

# horseshoe hyper-parameters
P0 = 2.0          # prior guess at the number of relevant covariates
PSEUDO_SIGMA = 2.0
SLAB_DF, SLAB_SCALE = 4.0, 2.0

OUTCOMES = [
    ("pub_nimby", "pub_n", "public NIMBY share"),
    ("pub_yimby", "pub_n", "public YIMBY share"),
    ("all_nimby", "all_n", "all-roles NIMBY share"),
    ("all_yimby", "all_n", "all-roles YIMBY share"),
]


def load():
    rows = []
    for r in csv.DictReader(open(OUT / "outcomes_clusters.csv")):
        d = dict(route=r["route"], body=r["body"], body_type=r["body_type"],
                 geo_kind=r["geo_kind"])
        for k in ("pub_n", "pub_nimby", "pub_yimby", "all_n", "all_nimby",
                  "all_yimby", "dais_n"):
            d[k] = int(r[k] or 0)
        for k in ("renter_share", "med_hh_income", "pop_density", "population",
                  "med_home_value", "med_gross_rent", "rent_burden_30",
                  "built_2010_plus", "multiunit_5plus", "ba_plus", "median_age",
                  "nhwhite_share", "transit_commute", "vacancy_rate"):
            d[k] = float(r[k])
        d["log_income"] = float(np.log(d["med_hh_income"]))
        d["log_density"] = float(np.log(d["pop_density"]))
        d["log_home_value"] = float(np.log(d["med_home_value"]))
        d["log_gross_rent"] = float(np.log(d["med_gross_rent"]))
        d["log_population"] = float(np.log(d["population"]))
        rows.append(d)
    return rows


def design(rows, cols):
    """Standardised covariate matrix + the raw sd of each column (so a
    coefficient can be put back on the covariate's own scale)."""
    X = np.column_stack([[r[c] for r in rows] for c in cols]).astype(float)
    mu, sd = X.mean(0), X.std(0, ddof=0)
    sd = np.where(sd > 0, sd, 1.0)
    return (X - mu) / sd, mu, sd


def summarise(idata, names):
    out = {}
    post = idata.posterior
    for n in names:
        if n not in post.data_vars:
            continue
        arr = np.asarray(post[n])
        if arr.ndim == 2:                      # scalar parameter
            cols = {n: arr.ravel()}
        else:
            cols = {f"{n}[{i}]": arr[..., i].ravel() for i in range(arr.shape[-1])}
        rh = az.rhat(idata, var_names=[n])[n].to_numpy()
        es = az.ess(idata, var_names=[n])[n].to_numpy()
        rh = np.atleast_1d(rh).ravel()
        es = np.atleast_1d(es).ravel()
        for i, (k, d) in enumerate(cols.items()):
            out[k] = dict(mean=float(d.mean()), median=float(np.median(d)),
                          sd=float(d.std(ddof=1)),
                          ci90=[float(np.percentile(d, 5)),
                                float(np.percentile(d, 95))],
                          ci95=[float(np.percentile(d, 2.5)),
                                float(np.percentile(d, 97.5))],
                          pr_gt_0=float((d > 0).mean()),
                          r_hat=float(rh[min(i, len(rh) - 1)]),
                          ess_bulk=float(es[min(i, len(es) - 1)]))
    return out


def diag(idata):
    skip = {"mu", "p", "eta", "gamma_route", "beta", "lambda_tilde"}
    names = [v for v in idata.posterior.data_vars if v not in skip]
    rhat = az.rhat(idata, var_names=names)
    ess = az.ess(idata, var_names=names)
    return dict(
        max_r_hat=float(max(float(np.nanmax(rhat[v].to_numpy())) for v in names)),
        min_ess_bulk=float(min(float(np.nanmin(ess[v].to_numpy())) for v in names)),
        divergences=(int(idata.sample_stats["diverging"].sum())
                     if "diverging" in idata.sample_stats else -1))


# --------------------------------------------------------------------- HC3
def ols_hc3(X, y, w=None):
    """OLS with HC3 (MacKinnon-White) standard errors. X includes no intercept;
    one is added. HC3 meat = diag(e_i^2 / (1 - h_i)^2)."""
    X = np.column_stack([X, np.ones(len(y))])
    y = np.asarray(y, float)
    XtXi = np.linalg.pinv(X.T @ X)
    beta = XtXi @ X.T @ y
    e = y - X @ beta
    h = np.einsum("ij,jk,ik->i", X, XtXi, X)
    omega = (e ** 2) / np.maximum((1 - h) ** 2, 1e-12)
    V = XtXi @ (X.T * omega) @ X @ XtXi
    se = np.sqrt(np.maximum(np.diag(V), 0))
    return beta, se


def logit_hc3(X, ycount, N, maxit=200, tol=1e-10):
    """Binomial logistic IRLS + an HC3-type sandwich.

    A = X' W X with W = diag(N p (1-p)); the score contribution of cell i is
    x_i (y_i - N_i p_i), so the robust meat is X' diag(s_i^2/(1-h_i)^2) X with
    h_i the IRLS leverage. Reduces to the usual HC3 in the Gaussian case.
    Returns (beta, se, converged)."""
    X = np.column_stack([X, np.ones(len(ycount))])
    ycount = np.asarray(ycount, float)
    N = np.asarray(N, float)
    beta = np.zeros(X.shape[1])
    ok = False
    for _ in range(maxit):
        eta = X @ beta
        p = 1 / (1 + np.exp(-np.clip(eta, -30, 30)))
        W = np.maximum(N * p * (1 - p), 1e-9)
        z = eta + (ycount - N * p) / W
        A = (X.T * W) @ X
        new = np.linalg.pinv(A) @ ((X.T * W) @ z)
        if np.max(np.abs(new - beta)) < tol:
            beta = new
            ok = True
            break
        beta = new
    eta = X @ beta
    p = 1 / (1 + np.exp(-np.clip(eta, -30, 30)))
    W = np.maximum(N * p * (1 - p), 1e-9)
    A = (X.T * W) @ X
    Ai = np.linalg.pinv(A)
    Xw = X * np.sqrt(W)[:, None]
    h = np.einsum("ij,jk,ik->i", Xw, np.linalg.pinv(Xw.T @ Xw), Xw)
    s = (ycount - N * p) ** 2 / np.maximum((1 - h) ** 2, 1e-12)
    V = Ai @ (X.T * s) @ X @ Ai
    return beta, np.sqrt(np.maximum(np.diag(V), 0)), ok


# ------------------------------------------------------------------- the fit
def fit(rows, ycol, ncol, cols, spec, label, with_u=True):
    y = np.array([r[ycol] for r in rows], int)
    N = np.array([r[ncol] for r in rows], int)
    Z, mu, sd = design(rows, cols)
    routes = sorted({r["route"] for r in rows}, key=lambda s: (s != ROUTE_REF, s))
    ridx = np.array([routes.index(r["route"]) for r in rows])
    n, p = len(rows), len(cols)

    with pm.Model():
        alpha = pm.Normal("alpha", 0, 1.5)
        if with_u:
            sigma_cell = pm.HalfNormal("sigma_cell", 0.75)
            u = pm.Normal("u", 0, sigma_cell, shape=n)
        else:
            u = 0.0
        if spec == "hs":
            tau0 = (P0 / max(p - P0, 1e-6)) * PSEUDO_SIGMA / np.sqrt(n)
            tau = pm.HalfCauchy("tau", tau0)
            lam = pm.HalfCauchy("lam", 1.0, shape=p)
            c2 = pm.InverseGamma("c2", SLAB_DF / 2.0,
                                 (SLAB_DF / 2.0) * SLAB_SCALE ** 2)
            lam_t = pt.sqrt(c2 * lam ** 2 / (c2 + tau ** 2 * lam ** 2))
            pm.Deterministic("lambda_tilde", lam_t)
            zb = pm.Normal("z_beta", 0, 1, shape=p)
            beta = pm.Deterministic("beta", zb * tau * lam_t)
        else:
            beta = pm.Normal("beta", 0, 1.0, shape=p)
        eta = alpha + pt.dot(Z, beta) + u
        if len(routes) > 1:
            g_raw = pm.Normal("gamma_route_raw", 0, 1.0, shape=len(routes) - 1)
            gamma = pm.Deterministic(
                "gamma_route", pt.concatenate([[0.0], g_raw]))
            eta = eta + gamma[ridx]
        pm.Binomial("y", n=N, p=pm.math.sigmoid(eta), observed=y)
        # The horseshoe funnel needs a tighter step size and longer adaptation
        # than the two Normal-prior specs; at 0.95/1500 it left 18-69
        # divergences per fit.
        ta = 0.99 if spec == "hs" else 0.95
        tn = TUNE_HS if spec == "hs" else TUNE
        idata = pm.sample(draws=DRAWS, tune=tn, chains=CHAINS,
                          target_accept=ta, random_seed=SEED,
                          progressbar=False,
                          idata_kwargs={"log_likelihood": True})
        pm.sample_posterior_predictive(idata, var_names=["y"],
                                       extend_inferencedata=True,
                                       random_seed=SEED, progressbar=False)

    names = (["alpha", "beta"] + (["sigma_cell"] if with_u else [])
             + (["tau"] if spec == "hs" else []))
    res = dict(label=label, spec=spec, with_u=with_u, outcome=ycol, denom=ncol,
               covariates=cols, n_cells=n, n_bodies=len({r["body"] for r in rows}),
               n_units=int(N.sum()), routes=routes,
               cells=[f"{r['route']}/{r['body']}" for r in rows],
               cov_mean=dict(zip(cols, map(float, mu))),
               cov_sd=dict(zip(cols, map(float, sd))),
               params=summarise(idata, names), diagnostics=diag(idata))

    # per-covariate convenience block, keyed by name
    b = np.asarray(idata.posterior["beta"])
    b = b.reshape(-1, p)
    res["beta_by_name"] = {}
    for j, c in enumerate(cols):
        d = b[:, j]
        res["beta_by_name"][c] = dict(
            median=float(np.median(d)),
            ci90=[float(np.percentile(d, 5)), float(np.percentile(d, 95))],
            ci95=[float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
            pr_gt_0=float((d > 0).mean()),
            excludes_zero_90=bool(np.percentile(d, 5) > 0 or np.percentile(d, 95) < 0),
            excludes_zero_95=bool(np.percentile(d, 2.5) > 0 or np.percentile(d, 97.5) < 0),
            per_10pt_renter=(float(np.median(d) * 0.10 / sd[j])
                             if c == "renter_share" else None))

    # LOO
    try:
        el = az.loo(idata, pointwise=True)
        khat = np.asarray(el.pareto_k) if hasattr(el, "pareto_k") else None
        res["loo"] = dict(elpd=float(el.elpd), se=float(el.se),
                          p_loo=float(getattr(el, "p", float("nan"))),
                          n_khat_gt_0p7=(int((khat > 0.7).sum())
                                         if khat is not None else None))
    except Exception as e:                                    # pragma: no cover
        res["loo"] = dict(error=repr(e))

    # PPC
    ppc = np.asarray(idata.posterior_predictive["y"]).reshape(-1, n)
    lo, hi = np.percentile(ppc, 5, axis=0), np.percentile(ppc, 95, axis=0)
    res["ppc_cells_inside_90"] = int(((y >= lo) & (y <= hi)).sum())

    # frequentist sanity checks on the same cells and the same covariates
    share = y / np.maximum(N, 1)
    ob, ose = ols_hc3(Z, share)
    lb, lse, conv = logit_hc3(Z, y, N)
    res["ols_hc3"] = {c: dict(beta=float(ob[j]), se=float(ose[j]),
                              lo95=float(ob[j] - 1.96 * ose[j]),
                              hi95=float(ob[j] + 1.96 * ose[j]))
                      for j, c in enumerate(cols)}
    res["logit_hc3"] = {c: dict(beta=float(lb[j]), se=float(lse[j]),
                                lo95=float(lb[j] - 1.96 * lse[j]),
                                hi95=float(lb[j] + 1.96 * lse[j]))
                        for j, c in enumerate(cols)}
    res["logit_hc3_converged"] = bool(conv)
    return res, idata


def main():
    MOUT.mkdir(parents=True, exist_ok=True)
    rows = load()
    screen = json.loads((OUT / "screen.json").read_text())
    kept = screen["kept"]
    base = ["renter_share", "log_income"]
    base_dens = base + ["log_density"]
    print(f"{STAMP}\nkept covariates (VIF screen): {kept}")

    SPECS = (("base", base, "(a) base: renter + income"),
             ("base_dens", base_dens, "(b) base + density"),
             ("hs", kept, "(c) full, regularised horseshoe"))

    results, idatas, idatas_nou, results_nou = [], {}, {}, []
    for ycol, ncol, nice in OUTCOMES:
        cells = [r for r in rows if r[ncol] >= MIN_N]
        print(f"\n=== {nice}: {len(cells)} cells, "
              f"{len({r['body'] for r in cells})} bodies, "
              f"{sum(r[ncol] for r in cells)} clusters")
        # Secondary fits WITHOUT the cell-level overdispersion term. These are
        # not the reported models -- the brief mandates overdispersion -- but
        # `u_i` gives every observation its own free parameter, which makes
        # PSIS-LOO degenerate (p_loo ~ n, most k-hat > 0.7). The no-`u`
        # binomial has a proper per-observation likelihood, so its LOO is the
        # only honest comparison of the three COVARIATE SETS.
        for spec, cols, tag in SPECS:
            rn, idn = fit(cells, ycol, ncol, cols,
                          "hs" if spec == "hs" else "normal",
                          f"{nice} — {tag} [no overdispersion]", with_u=False)
            rn["model_key"], rn["outcome_nice"] = spec, nice
            results_nou.append(rn)
            idatas_nou[(ycol, spec)] = idn
        for spec, cols, tag in SPECS:
            label = f"{nice} — {tag}"
            r, idata = fit(cells, ycol, ncol, cols,
                           "hs" if spec == "hs" else "normal", label)
            r["model_key"] = spec
            r["outcome_nice"] = nice
            results.append(r)
            idatas[(ycol, spec)] = idata
            bn = r["beta_by_name"]["renter_share"]
            print(f"  {tag:34s} beta_renter {bn['median']:+.3f} "
                  f"90% [{bn['ci90'][0]:+.3f},{bn['ci90'][1]:+.3f}] "
                  f"Pr>0 {bn['pr_gt_0']:.2f}  "
                  f"R-hat {r['diagnostics']['max_r_hat']:.4f} "
                  f"ESS {r['diagnostics']['min_ess_bulk']:.0f} "
                  f"div {r['diagnostics']['divergences']} "
                  f"LOO {r['loo'].get('elpd', float('nan')):.2f}")

    # ---- LOO comparison within each outcome, for BOTH model families.
    # arviz 1.x `compare` returns rank / elpd_diff / dse / p_worse / diag_diff
    # / weight and does NOT carry an elpd_loo column, so the absolute ELPD and
    # its SE come from each fit's own az.loo.
    def loo_block(family, store, res_store):
        out = []
        for ycol, _ncol, nice in OUTCOMES:
            d = {k: store[(ycol, k)] for k in ("base", "base_dens", "hs")
                 if (ycol, k) in store}
            own = {k: next(x for x in res_store
                           if x["outcome"] == ycol and x["model_key"] == k)
                   for k in d}
            def rec_for(name, r, rank, diff, dse, pw, dd, wt):
                return dict(family=family, outcome=nice, model=name, rank=rank,
                            elpd_loo=r["loo"].get("elpd", float("nan")),
                            loo_se=r["loo"].get("se", float("nan")),
                            p_loo=r["loo"].get("p_loo", float("nan")),
                            elpd_diff=diff, dse=dse, p_worse=pw, diag_diff=dd,
                            weight=wt,
                            n_khat_gt_0p7=r["loo"].get("n_khat_gt_0p7"),
                            n_obs=r["n_cells"],
                            divergences=r["diagnostics"]["divergences"])
            try:
                df = az.compare(d, round_to=None)
                for name, rec in df.iterrows():
                    out.append(rec_for(
                        name, own[name], int(rec["rank"]),
                        float(rec["elpd_diff"]), float(rec["dse"]),
                        float(rec["p_worse"]) if "p_worse" in rec else float("nan"),
                        str(rec.get("diag_diff", "")), float(rec["weight"])))
            except Exception as e:
                print(f"az.compare failed for {family}/{nice}: {e!r}; raw ELPD only")
                for k, r in own.items():
                    out.append(rec_for(k, r, -1, float("nan"), float("nan"),
                                       float("nan"), "compare failed",
                                       float("nan")))
        return out

    loo_rows = (loo_block("reported (cell-level overdispersion)", idatas, results)
                + loo_block("no-overdispersion (LOO-valid)", idatas_nou,
                            results_nou))

    with open(MOUT / "loo_table.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(loo_rows[0]))
        w.writeheader()
        for r in loo_rows:
            w.writerow(r)

    with open(MOUT / "model_table.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["outcome", "model", "covariate", "n_cells", "n_bodies",
                    "n_clusters", "beta_median", "ci90_lo", "ci90_hi",
                    "ci95_lo", "ci95_hi", "pr_gt_0", "excl_zero_90",
                    "max_r_hat", "min_ess", "divergences", "elpd_loo",
                    "ppc_inside_90"])
        for r in results:
            for c in r["covariates"]:
                b = r["beta_by_name"][c]
                w.writerow([r["outcome_nice"], r["model_key"], c, r["n_cells"],
                            r["n_bodies"], r["n_units"],
                            f"{b['median']:+.4f}", f"{b['ci90'][0]:+.4f}",
                            f"{b['ci90'][1]:+.4f}", f"{b['ci95'][0]:+.4f}",
                            f"{b['ci95'][1]:+.4f}", f"{b['pr_gt_0']:.3f}",
                            int(b["excludes_zero_90"]),
                            f"{r['diagnostics']['max_r_hat']:.4f}",
                            f"{r['diagnostics']['min_ess_bulk']:.0f}",
                            r["diagnostics"]["divergences"],
                            f"{r['loo'].get('elpd', float('nan')):.3f}",
                            f"{r['ppc_cells_inside_90']}/{r['n_cells']}"])

    with open(MOUT / "ols_hc3.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["outcome", "model", "covariate", "ols_beta", "ols_hc3_se",
                    "ols_lo95", "ols_hi95", "logit_beta", "logit_hc3_se",
                    "logit_lo95", "logit_hi95", "logit_converged"])
        for r in results:
            for c in r["covariates"]:
                o, l = r["ols_hc3"][c], r["logit_hc3"][c]
                w.writerow([r["outcome_nice"], r["model_key"], c,
                            f"{o['beta']:+.4f}", f"{o['se']:.4f}",
                            f"{o['lo95']:+.4f}", f"{o['hi95']:+.4f}",
                            f"{l['beta']:+.4f}", f"{l['se']:.4f}",
                            f"{l['lo95']:+.4f}", f"{l['hi95']:+.4f}",
                            int(r["logit_hc3_converged"])])

    (MOUT / "model_results.json").write_text(json.dumps(dict(
        stamp=STAMP, seed=SEED, draws=DRAWS, tune=TUNE, chains=CHAINS,
        floor_clusters=MIN_N, route_reference=ROUTE_REF,
        horseshoe=dict(p0=P0, pseudo_sigma=PSEUDO_SIGMA, slab_df=SLAB_DF,
                       slab_scale=SLAB_SCALE),
        kept_covariates=kept, screen=screen,
        results=results, results_no_overdispersion=results_nou,
        loo=loo_rows,
        loo_note=("PSIS-LOO on the reported models is degenerate: the "
                  "cell-level term u_i is one free parameter per observation, "
                  "so p_loo approaches n and most Pareto k-hat exceed 0.7. "
                  "The `no-overdispersion (LOO-valid)` block compares the "
                  "same three covariate sets in a plain binomial, where the "
                  "per-observation likelihood is proper. Neither block is "
                  "decision-grade at n = 29-35.")), indent=2, default=str))

    print("\n--- LOO ranking")
    fam = None
    for r in loo_rows:
        if r["family"] != fam:
            fam = r["family"]
            print(f"  [{fam}]")
        print(f"  {r['outcome']:24s} {r['model']:10s} rank {r['rank']:2d} "
              f"elpd {r['elpd_loo']:+9.3f} (se {r['loo_se']:5.2f}) "
              f"diff {r['elpd_diff']:+7.3f} dse {r['dse']:6.3f} "
              f"p_loo {r['p_loo']:5.2f} khat>0.7 {r['n_khat_gt_0p7']}/{r['n_obs']} "
              f"div {r['divergences']} {r['diag_diff']}")
    print("\n--- covariates whose 90 % excludes zero anywhere")
    any_sig = False
    for r in results:
        for c in r["covariates"]:
            b = r["beta_by_name"][c]
            if b["excludes_zero_90"]:
                any_sig = True
                print(f"  {r['outcome_nice']:24s} {r['model_key']:10s} "
                      f"{c:18s} {b['median']:+.3f} "
                      f"90% [{b['ci90'][0]:+.3f},{b['ci90'][1]:+.3f}] "
                      f"Pr>0 {b['pr_gt_0']:.3f}")
    if not any_sig:
        print("  none")
    print(f"\n-> {MOUT}")


if __name__ == "__main__":
    main()
