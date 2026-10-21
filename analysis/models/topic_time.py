#!/usr/bin/env python3
"""
topic_time.py -- estimand 1 (plan/SHOWCASE.md 1.1): episode-minutes per
meeting-hour by body x month x family, as a negative-binomial rate model with
body random intercepts, a month random walk per family and the locate route as
a stratum. Model statement and priors: analysis/models/README.md 1.

  y_i ~ NegBin(mu_i, phi)
  log mu_i = log E_i + alpha + beta_f(i) + u_b(i) + v_b(i),f(i) + gamma_r(i) + w_f(i),t(i)

FACET. fit() takes ONE cell table. A table spanning routes is refused unless it
was built with load.pool_routes(pool=True) (its `pooled` column says so) and
then the route enters as gamma_r; a per-route table (one route) drops gamma.
fit_per_route() is the per-facet result; the pooled fit is the explicitly
stamped one. Nothing here pools by accident.

    .venv/bin/python analysis/models/topic_time.py --out output/analysis/models
"""
import argparse, sys, time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from analysis.models import load as L  # noqa: E402

MIN_CELLS = 12         # under this a per-route fit is a prior with a data footnote
CI = (5, 95)           # 90 % intervals throughout (SHOWCASE.md 1)
PRIORS = {             # one line each, README 1.3
    "alpha": ("Normal(1, 1.5)", "mean log minutes per meeting-hour; e^1 = 2.7 min/h is about T2's expectation of 5 % topic time, sd 1.5 spans 0.2 to 30 min/h"),
    "beta": ("Normal(0, 1.5), sum-to-zero over families", "a family can differ from the mean by a factor of e^1.5 = 4.5 at one sd; no direction assumed"),
    "sigma_b": ("HalfNormal(1)", "between-body sd of log rate; 95 % mass under a 7-fold ratio-sd, which is wider than any plausible council-to-council spread"),
    "sigma_v": ("HalfNormal(0.5)", "body x family interaction sd, half the main-effect scale: a body's family mix can differ, but less than bodies differ"),
    "gamma": ("Normal(0, 1.5), reference = largest route", "route recall differences enter as a stratum, not a prior claim about direction"),
    "sigma_w": ("HalfNormal(0.3)", "month-to-month sd of the log rate random walk; 0.3 is a 35 % monthly change at one sd, generous for a 13-month window"),
    "phi": ("1/phi ~ Exponential(1)", "inverse dispersion with most mass near 0 (Poisson) and a long tail, so overdispersion is learned, not assumed"),
    "zi_a": ("Normal(1, 1.5)", "logit P(not a structural zero) at mean exposure; centred on 73 %, which is the observed non-zero share, sd wide enough to reach 10-99 %"),
    "zi_b": ("Normal(0, 1)", "slope of that logit in centred log exposure; one log-hour of extra exposure may move it by a factor e at one sd"),
}


# ------------------------------------------------------------------ fit ------
def _index(values):
    levels = sorted(pd.unique(values))
    return levels, np.array([levels.index(v) for v in values])


def design(cells):
    """Integer indices and level lists for one cell table."""
    bodies, b = _index(cells["body"])
    fams, f = _index(cells["family"])
    months, t = _index(cells["month"])
    routes = sorted(cells["route"].unique(), key=lambda r: (-(cells["route"] == r).sum(), r))  # largest first
    r = np.array([routes.index(v) for v in cells["route"]])
    return {"bodies": bodies, "b": b, "families": fams, "f": f, "months": months, "t": t,
            "routes": routes, "r": r, "E": cells["exposure_h"].to_numpy(float), "y": cells["y"].to_numpy(int)}


def fit(cells, draws=1000, tune=1000, chains=4, seed=20261017, target_accept=0.9, progress=False,
        body_effects=None, month_walk=None, likelihood="zinb"):
    """One NB fit on one cell table. Refuses a multi-route table that was not
    pooled on purpose (load.pool_routes). Returns a dict with the InferenceData,
    the design, posterior summaries and diagnostics."""
    import pymc as pm
    cells = cells.reset_index(drop=True)
    n_routes = cells["route"].nunique()
    pooled = bool(cells["pooled"].any()) if "pooled" in cells.columns else False
    if n_routes > 1 and not pooled:
        raise L.FacetError("topic_time.fit: cell table spans %d locate routes and was not built with "
                           "pool_routes(pool=True); fit per route or pool on purpose" % n_routes)
    d = design(cells)
    B, F, T, R = len(d["bodies"]), len(d["families"]), len(d["months"]), len(d["routes"])
    if body_effects is None:
        body_effects = B >= 2
    if month_walk is None:
        month_walk = T >= 3
    with pm.Model(coords={"body": d["bodies"], "family": d["families"], "month": d["months"], "route": d["routes"]}) as model:
        alpha = pm.Normal("alpha", 1.0, 1.5)
        beta_raw = pm.Normal("beta_raw", 0.0, 1.5, dims="family")
        beta = pm.Deterministic("beta", beta_raw - beta_raw.mean(), dims="family")
        eta = alpha + beta[d["f"]]
        if body_effects:
            sigma_b = pm.HalfNormal("sigma_b", 1.0)
            z_b = pm.Normal("z_b", 0.0, 1.0, dims="body")
            u = pm.Deterministic("u", sigma_b * z_b, dims="body")
            sigma_v = pm.HalfNormal("sigma_v", 0.5)
            z_v = pm.Normal("z_v", 0.0, 1.0, dims=("body", "family"))
            v = pm.Deterministic("v", sigma_v * z_v, dims=("body", "family"))
            eta = eta + u[d["b"]] + v[d["b"], d["f"]]
        if R > 1:
            gamma_free = pm.Normal("gamma_free", 0.0, 1.5, shape=R - 1)
            gamma = pm.Deterministic("gamma", pm.math.concatenate([[0.0], gamma_free]), dims="route")
            eta = eta + gamma[d["r"]]
        if month_walk:
            sigma_w = pm.HalfNormal("sigma_w", 0.3)
            z_w = pm.Normal("z_w", 0.0, 1.0, shape=(F, T - 1))
            steps = pm.math.cumsum(sigma_w * z_w, axis=1)
            walk = pm.math.concatenate([pm.math.zeros((F, 1)), steps], axis=1)
            w = pm.Deterministic("w", walk - walk.mean(axis=1, keepdims=True), dims=("family", "month"))
            eta = eta + w[d["f"], d["t"]]
        inv_phi = pm.Exponential("inv_phi", 1.0)
        phi = pm.Deterministic("phi", 1.0 / inv_phi)
        mu = pm.Deterministic("mu", pm.math.exp(np.log(d["E"]) + eta))
        if likelihood == "nb":
            pm.NegativeBinomial("y", mu=mu, alpha=phi, observed=d["y"])
        else:
            # zero inflation: a body-month the route read with no located episode
            # at all is a different event from a short one. psi = P(not a
            # structural zero), logit-linear in log exposure: a 40-hour month is
            # less likely to have nothing than a 2-hour one.
            logE_c = np.log(d["E"]) - np.log(d["E"]).mean()
            zi_a = pm.Normal("zi_a", 1.0, 1.5)
            zi_b = pm.Normal("zi_b", 0.0, 1.0)
            psi = pm.Deterministic("psi", pm.math.sigmoid(zi_a + zi_b * logE_c))
            pm.Deterministic("psi_bar", pm.math.sigmoid(zi_a))   # P(active) at mean exposure
            pm.ZeroInflatedNegativeBinomial("y", psi=psi, mu=mu, alpha=phi, observed=d["y"])
        if body_effects:
            pm.Deterministic("body_rate", pm.math.exp(alpha + u), dims="body")
            pm.Deterministic("body_family_rate", pm.math.exp(alpha + beta[None, :] + u[:, None] + v), dims=("body", "family"))
        if month_walk:
            pm.Deterministic("month_rate", pm.math.exp(alpha + beta[:, None] + w), dims=("family", "month"))
        t0 = time.time()
        idata = pm.sample(draws=draws, tune=tune, chains=chains, cores=min(chains, 4), random_seed=seed,
                          target_accept=target_accept, progressbar=progress)
        secs = time.time() - t0
        pm.sample_posterior_predictive(idata, var_names=["y"], extend_inferencedata=True, random_seed=seed, progressbar=progress)
    out = {"idata": idata, "design": d, "cells": cells, "pooled": pooled, "seconds": secs,
           "body_effects": body_effects, "month_walk": month_walk, "likelihood": likelihood,
           "settings": {"draws": draws, "tune": tune, "chains": chains, "target_accept": target_accept, "seed": seed,
                        "likelihood": likelihood}}
    out["diagnostics"] = diagnostics(idata)
    return out


# ------------------------------------------------------------ summaries ------
def _q(x, axis=0):
    lo, hi = np.percentile(x, CI, axis=axis)
    return np.median(x, axis=axis), lo, hi


def _flat(idata, name):
    da = idata.posterior[name]
    return da.stack(sample=("chain", "draw")).transpose("sample", ...).to_numpy()


def diagnostics(idata):
    import arviz as az
    skip = {"mu", "psi", "z_b", "z_v", "z_w", "beta_raw", "gamma_free", "body_family_rate", "month_rate", "body_rate"}
    names = [v for v in idata.posterior.data_vars if v not in skip]
    rhat = az.rhat(idata, var_names=names)
    ess = az.ess(idata, var_names=names)
    rh = float(max(float(np.nanmax(rhat[v].to_numpy())) for v in names))
    es = float(min(float(np.nanmin(ess[v].to_numpy())) for v in names))
    div = int(idata.sample_stats["diverging"].sum()) if "diverging" in idata.sample_stats else -1
    n = int(idata.posterior.sizes["chain"] * idata.posterior.sizes["draw"])
    return {"max_rhat": round(rh, 4), "min_ess_bulk": round(es, 0), "divergences": div, "draws_total": n,
            "ok": bool(rh < 1.01 and div == 0 and es >= 400)}


def body_table(fit_):
    """Posterior body rate (min per meeting-hour, family-mean, reference route,
    month-average) next to the crude rate, ranked."""
    if not fit_["body_effects"]:
        return pd.DataFrame()
    d, cells = fit_["design"], fit_["cells"]
    br = _flat(fit_["idata"], "body_rate")
    med, lo, hi = _q(br)
    raw = L.raw_rates(cells, by=("body",)).set_index("body")
    rows = []
    for i, b in enumerate(d["bodies"]):
        rows.append({"body": b, "post_median": med[i], "lo90": lo[i], "hi90": hi[i],
                     "raw_rate": raw.loc[b, "raw_rate"], "minutes": raw.loc[b, "minutes"],
                     "exposure_h": raw.loc[b, "exposure_h"], "n_cells": int(raw.loc[b, "n_cells"]),
                     "pr_above_median_body": float((br[:, i] > np.median(br, axis=1)).mean())})
    df = pd.DataFrame(rows).sort_values("post_median", ascending=False).reset_index(drop=True)
    df.insert(1, "rank", np.arange(1, len(df) + 1))
    if "psi_bar" in fit_["idata"].posterior:
        # the ZINB body_rate is conditional on an active cell; the marginal
        # expectation at mean exposure multiplies by P(active)
        psi = _flat(fit_["idata"], "psi_bar")
        marg = br * psi[:, None]
        mm, ml, mh = _q(marg)
        idx = {b: i for i, b in enumerate(d["bodies"])}
        df["marginal_median"] = [mm[idx[b]] for b in df["body"]]
        df["marginal_lo90"] = [ml[idx[b]] for b in df["body"]]
        df["marginal_hi90"] = [mh[idx[b]] for b in df["body"]]
    # family-specific rates alongside
    bfr = _flat(fit_["idata"], "body_family_rate")
    for j, f in enumerate(d["families"]):
        m, l, h = _q(bfr[:, :, j])
        fam = pd.DataFrame({"body": d["bodies"], f"{f}_median": m, f"{f}_lo90": l, f"{f}_hi90": h})
        df = df.merge(fam, on="body", how="left")
    return df


def contrast_table(fit_):
    """transit vs density: rate ratio exp(beta_transit - beta_density) and the
    absolute difference in minutes per meeting-hour at the mean body."""
    d = fit_["design"]
    beta = _flat(fit_["idata"], "beta")
    alpha = _flat(fit_["idata"], "alpha")
    rows = []
    fams = d["families"]
    for i, fi in enumerate(fams):
        for j, fj in enumerate(fams):
            if j <= i:
                continue
            lr = beta[:, j] - beta[:, i]
            ratio = np.exp(lr)
            diff = np.exp(alpha + beta[:, j]) - np.exp(alpha + beta[:, i])
            m, l, h = _q(ratio); dm, dl, dh = _q(diff)
            rows.append({"contrast": f"{fj} / {fi}", "ratio_median": m, "ratio_lo90": l, "ratio_hi90": h,
                         "pr_ratio_gt_1": float((ratio > 1).mean()),
                         "diff_min_per_h_median": dm, "diff_lo90": dl, "diff_hi90": dh})
    for i, f in enumerate(fams):
        r = np.exp(alpha + beta[:, i]); m, l, h = _q(r)
        rows.append({"contrast": f"{f} rate (mean body)", "ratio_median": m, "ratio_lo90": l, "ratio_hi90": h,
                     "pr_ratio_gt_1": np.nan, "diff_min_per_h_median": np.nan, "diff_lo90": np.nan, "diff_hi90": np.nan})
    return pd.DataFrame(rows)


def month_table(fit_):
    if not fit_["month_walk"]:
        return pd.DataFrame()
    d = fit_["design"]
    mr = _flat(fit_["idata"], "month_rate")
    rows = []
    cells = fit_["cells"]
    for i, f in enumerate(d["families"]):
        m, l, h = _q(mr[:, i, :])
        for j, mo in enumerate(d["months"]):
            c = cells[(cells["family"] == f) & (cells["month"] == mo)]
            rows.append({"family": f, "month": mo, "rate_median": m[j], "lo90": l[j], "hi90": h[j],
                         "raw_rate": c["minutes"].sum() / c.drop_duplicates(["body", "route"])["exposure_h"].sum() if len(c) else np.nan,
                         "n_cells": len(c), "exposure_h": c.drop_duplicates(["body", "route"])["exposure_h"].sum()})
    return pd.DataFrame(rows)


def route_table(fit_):
    d = fit_["design"]
    if len(d["routes"]) < 2:
        return pd.DataFrame()
    g = _flat(fit_["idata"], "gamma")
    m, l, h = _q(np.exp(g))
    return pd.DataFrame({"route": d["routes"], "rate_ratio_vs_reference": m, "lo90": l, "hi90": h,
                         "n_cells": [int((fit_["cells"]["route"] == r).sum()) for r in d["routes"]],
                         "reference": [r == d["routes"][0] for r in d["routes"]]})


def hyper_table(fit_):
    rows = []
    for name in ("alpha", "sigma_b", "sigma_v", "sigma_w", "phi", "zi_a", "zi_b"):
        if name in fit_["idata"].posterior:
            x = _flat(fit_["idata"], name); m, l, h = _q(x)
            rows.append({"parameter": name, "median": m, "lo90": l, "hi90": h, "prior": PRIORS.get(name, ("", ""))[0]})
    return pd.DataFrame(rows)


def ppc_table(fit_):
    """Observed vs replicated: zero share, total minutes, max cell, per route and family."""
    y = fit_["design"]["y"]
    rep = fit_["idata"].posterior_predictive["y"].stack(sample=("chain", "draw")).transpose("sample", ...).to_numpy()
    cells = fit_["cells"]
    rows = []

    def add(label, mask):
        obs, r = y[mask], rep[:, mask]
        for stat, fo, fr in (("zero_share", (obs == 0).mean(), (r == 0).mean(axis=1)),
                             ("total_minutes", obs.sum(), r.sum(axis=1)),
                             ("max_cell", obs.max(), r.max(axis=1)),
                             ("p90_cell", np.percentile(obs, 90), np.percentile(r, 90, axis=1))):
            lo, hi = np.percentile(fr, CI)
            rows.append({"subset": label, "stat": stat, "observed": float(fo), "rep_median": float(np.median(fr)),
                         "rep_lo90": float(lo), "rep_hi90": float(hi),
                         "p_rep_ge_obs": float((fr >= fo).mean()), "n_cells": int(mask.sum())})
    add("all", np.ones(len(y), bool))
    for f in sorted(cells["family"].unique()):
        add(f"family={f}", (cells["family"] == f).to_numpy())
    for r in sorted(cells["route"].unique()):
        add(f"route={r}", (cells["route"] == r).to_numpy())
    return pd.DataFrame(rows)


def summarise(fit_):
    return {"bodies": body_table(fit_), "contrast": contrast_table(fit_), "months": month_table(fit_),
            "routes": route_table(fit_), "hyper": hyper_table(fit_), "ppc": ppc_table(fit_)}


# ------------------------------------------------------- per-route sweep ------
def fit_per_route(cells, min_cells=MIN_CELLS, **kw):
    """One fit per locate route, never pooled. Routes with fewer than min_cells
    cells are reported, not fitted."""
    out, skipped = {}, {}
    for r in sorted(cells["route"].unique()):
        c = cells[cells["route"] == r].copy()
        c["pooled"] = False
        if len(c) < min_cells:
            skipped[r] = {"n_cells": int(len(c)), "reason": f"fewer than {min_cells} cells"}
            continue
        out[r] = fit(c, **kw)
    return out, skipped


# ------------------------------------------------------------------ main ------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(ROOT / "output" / "analysis" / "models"))
    ap.add_argument("--state", default=None)
    ap.add_argument("--draws", type=int, default=1000)
    ap.add_argument("--tune", type=int, default=1000)
    ap.add_argument("--chains", type=int, default=4)
    ap.add_argument("--no-sensitivity", action="store_true")
    a = ap.parse_args(argv)
    from analysis.models import run
    return run.topic_time_stage(Path(a.out), state=a.state, draws=a.draws, tune=a.tune, chains=a.chains,
                                sensitivity=not a.no_sensitivity)


if __name__ == "__main__":
    sys.exit(main())
