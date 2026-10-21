#!/usr/bin/env python3
"""
renter_share_models.py -- the two regressions behind plan/RENTER-SHARE.md.

n (bodies) is 15-25 at best and 7 at worst, so everything here is wide by
construction. Read the intervals, not the medians.

MODEL 1 -- public NIMBY share on renter share (binomial GLM, PyMC).
  Unit: a body x route cell. y = NIMBY count among public turns with a
  numeric stance in density/both-family episodes; N = that denominator.

    y_i   ~ Binomial(N_i, p_i)
    logit p_i = alpha + beta_renter * z_renter_i + gamma_route(i) + u_i
    u_i   ~ Normal(0, sigma_cell)        # cell-level overdispersion
    alpha ~ Normal(0, 1.5)   beta ~ Normal(0, 1)   gamma ~ Normal(0, 1)
    sigma_cell ~ HalfNormal(0.75)

  `u_i` is the body-level overdispersion term the brief asks for: with one
  cell per body x route it absorbs exactly the extra-binomial spread that a
  pure binomial would read as signal. Route enters as a fixed-effect stratum
  (reference = classifier, the largest), never pooled away. z_renter is
  renter share standardised over the fitted cells, so beta is "log-odds per
  1 sd of renter share"; the per-10-point slope is reported too.
  Variants: +controls (log income, log density), and a density-family-only
  sanity fit. Plus an unweighted and an N-weighted OLS on the shares.

MODEL 2 -- located topic minutes on renter share (NB with exposure).
  Unit: body x route x family cell from output/wave1/rates_by_body_route.csv.

    y_i ~ NegBinomial(mu_i, phi)
    log mu_i = log E_i + alpha + beta_fam(i) + beta_renter * z_renter_i
               + gamma_route(i) + u_body(i)
    u_body ~ Normal(0, sigma_b)

  Same shape and the same priors as analysis/models/topic_time.py (README 1)
  with the month walk, the body x family interaction and the zero-inflation
  dropped -- there are not enough cells here to identify them alongside a
  body-level covariate -- and renter share added at the body level. This is a
  reduced re-fit for one coefficient, not a replacement for `make models`.
  Primary fit = complete-coverage cells only; a sensitivity adds the partial
  bodies (whose rates are floors) with a `partial` indicator.

Reports posterior mean/median, 90 % and 95 % HDIs, Pr(beta > 0), R-hat, ESS,
divergences and PPCs. Writes output/renter_share/models/.
"""
import csv
import json
import pathlib

import numpy as np
import pymc as pm
import arviz as az
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "output" / "renter_share"
MOUT = OUT / "models"
STAMP = "EXPLORATORY — stance κ gate not passed (plan/SHOWCASE.md 4a)"
SEED = 20261018
DRAWS, TUNE, CHAINS = 2000, 1500, 4
MIN_N = 5
ROUTE_REF = "classifier"


def z(v):
    v = np.asarray(v, float)
    return (v - v.mean()) / v.std(ddof=0), float(v.mean()), float(v.std(ddof=0))


def summarise(idata, names):
    """Posterior summaries straight off the draws (arviz 1.x has no
    hdi_prob= on summary); intervals are equal-tailed percentile
    intervals, which is what every other table in this repo reports."""
    out = {}
    post = idata.posterior
    for n in names:
        if n not in post.data_vars:
            continue
        d = np.asarray(post[n]).ravel()
        rh = float(np.nanmax(az.rhat(idata, var_names=[n])[n].to_numpy()))
        es = float(np.nanmin(az.ess(idata, var_names=[n])[n].to_numpy()))
        out[n] = dict(mean=float(d.mean()), median=float(np.median(d)),
                      sd=float(d.std(ddof=1)),
                      hdi90=[float(np.percentile(d, 5)), float(np.percentile(d, 95))],
                      hdi95=[float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
                      pr_gt_0=float((d > 0).mean()),
                      r_hat=rh, ess_bulk=es)
    return out


def diag(idata):
    skip = {"mu", "p", "phi", "gamma_route", "beta_fam"}
    names = [v for v in idata.posterior.data_vars if v not in skip]
    rhat = az.rhat(idata, var_names=names)
    ess = az.ess(idata, var_names=names)
    return dict(
        max_r_hat=float(max(float(np.nanmax(rhat[v].to_numpy())) for v in names)),
        min_ess_bulk=float(min(float(np.nanmin(ess[v].to_numpy())) for v in names)),
        divergences=(int(idata.sample_stats["diverging"].sum())
                     if "diverging" in idata.sample_stats else -1))


def ols(xcols, y, w=None):
    X = np.column_stack(list(xcols) + [np.ones(len(y))])
    y = np.asarray(y, float)
    if w is None:
        w = np.ones(len(y))
    W = np.sqrt(np.asarray(w, float))
    beta, *_ = np.linalg.lstsq(X * W[:, None], y * W, rcond=None)
    res = y - X @ beta
    dof = len(y) - X.shape[1]
    out = dict(slope=float(beta[0]), intercept=float(beta[-1]), n=len(y), dof=dof)
    if dof > 0:
        s2 = float((w * res ** 2).sum() / dof)
        se = float(np.sqrt(s2 * np.linalg.pinv((X * W[:, None]).T @ (X * W[:, None]))[0, 0]))
        out.update(se=se, lo95=float(beta[0] - 1.96 * se),
                   hi95=float(beta[0] + 1.96 * se),
                   t=float(beta[0] / se) if se else None)
    return out


# ------------------------------------------------------------------- model 1
def fit_binomial(rows, label, controls=False):
    nimby = np.array([r["nimby_count"] for r in rows], int)
    N = np.array([r["public_turns_numeric"] for r in rows], int)
    zr, rmu, rsd = z([r["renter_share"] for r in rows])
    routes = sorted({r["route"] for r in rows},
                    key=lambda s: (s != ROUTE_REF, s))
    ridx = np.array([routes.index(r["route"]) for r in rows])
    nr = len(routes)
    extra = {}
    if controls:
        zi, *_ = z([np.log(r["med_hh_income"]) for r in rows])
        zd, *_ = z([np.log(r["pop_density"]) for r in rows])
        extra = dict(zi=zi, zd=zd)

    with pm.Model() as m:
        alpha = pm.Normal("alpha", 0, 1.5)
        beta_renter = pm.Normal("beta_renter", 0, 1.0)
        sigma_cell = pm.HalfNormal("sigma_cell", 0.75)
        u = pm.Normal("u", 0, sigma_cell, shape=len(rows))
        eta = alpha + beta_renter * zr + u
        if nr > 1:
            gamma_raw = pm.Normal("gamma_route_raw", 0, 1.0, shape=nr - 1)
            gamma = pm.Deterministic(
                "gamma_route", pm.math.concatenate([[0.0], gamma_raw]))
            eta = eta + gamma[ridx]
        if controls:
            b_inc = pm.Normal("beta_log_income", 0, 1.0)
            b_den = pm.Normal("beta_log_density", 0, 1.0)
            eta = eta + b_inc * extra["zi"] + b_den * extra["zd"]
        p = pm.Deterministic("p", pm.math.sigmoid(eta))
        pm.Binomial("y", n=N, p=p, observed=nimby)
        idata = pm.sample(draws=DRAWS, tune=TUNE, chains=CHAINS,
                          target_accept=0.95, random_seed=SEED,
                          progressbar=False, idata_kwargs={"log_likelihood": False})
        pm.sample_posterior_predictive(
            idata, var_names=["y"], extend_inferencedata=True,
            random_seed=SEED, progressbar=False)

    names = ["alpha", "beta_renter", "sigma_cell"]
    if controls:
        names += ["beta_log_income", "beta_log_density"]
    res = dict(label=label, n_cells=len(rows), n_turns=int(N.sum()),
               routes=routes, renter_mean=rmu, renter_sd=rsd,
               bodies=[f"{r['route']}/{r['body']}" for r in rows],
               params=summarise(idata, names), diagnostics=diag(idata))
    d = np.asarray(idata.posterior["beta_renter"]).ravel()
    res["beta_per_10pt_renter"] = dict(
        median=float(np.median(d) * 0.10 / rsd),
        hdi90=[float(np.percentile(d, 5) * 0.10 / rsd),
               float(np.percentile(d, 95) * 0.10 / rsd)],
        hdi95=[float(np.percentile(d, 2.5) * 0.10 / rsd),
               float(np.percentile(d, 97.5) * 0.10 / rsd)])
    # PPC: observed share inside the replicated 90 % band per cell?
    ppc = np.asarray(idata.posterior_predictive["y"]).reshape(-1, len(rows))
    lo, hi = np.percentile(ppc, 5, axis=0), np.percentile(ppc, 95, axis=0)
    res["ppc"] = dict(
        cells_inside_90=int(((nimby >= lo) & (nimby <= hi)).sum()),
        cells=len(rows),
        per_cell=[dict(cell=f"{r['route']}/{r['body']}", obs=int(nimby[i]),
                       N=int(N[i]), rep_lo90=float(lo[i]), rep_hi90=float(hi[i]))
                  for i, r in enumerate(rows)])
    # OLS sanity checks on the share
    sh = nimby / N
    res["ols_unweighted"] = ols([[r["renter_share"] for r in rows]], sh)
    res["ols_n_weighted"] = ols([[r["renter_share"] for r in rows]], sh, w=N)
    res["ols_with_controls"] = ols(
        [[r["renter_share"] for r in rows],
         [np.log(r["med_hh_income"]) for r in rows],
         [np.log(r["pop_density"]) for r in rows]], sh)
    return res, idata


# ------------------------------------------------------------------- model 2
def fit_nb(cells, label, with_partial=False):
    y = np.array([round(c["episode_minutes"]) for c in cells], int)
    E = np.array([c["meeting_hours"] for c in cells], float)
    zr, rmu, rsd = z([c["renter_share"] for c in cells])
    fams = sorted({c["family"] for c in cells})
    fidx = np.array([fams.index(c["family"]) for c in cells])
    routes = sorted({c["route"] for c in cells}, key=lambda s: (s != ROUTE_REF, s))
    ridx = np.array([routes.index(c["route"]) for c in cells])
    bodies = sorted({c["body"] for c in cells})
    bidx = np.array([bodies.index(c["body"]) for c in cells])
    part = np.array([0.0 if c["coverage"] == "complete" else 1.0 for c in cells])

    with pm.Model() as m:
        alpha = pm.Normal("alpha", 1.0, 1.5)
        if len(fams) > 1:
            bf_raw = pm.Normal("beta_fam_raw", 0, 1.5, shape=len(fams) - 1)
            beta_fam = pm.Deterministic(
                "beta_fam", pm.math.concatenate([bf_raw, [-pm.math.sum(bf_raw)]]))
            fam_term = beta_fam[fidx]
        else:
            fam_term = 0.0
        beta_renter = pm.Normal("beta_renter", 0, 1.0)
        sigma_b = pm.HalfNormal("sigma_b", 1.0)
        u = pm.Normal("u_body", 0, sigma_b, shape=len(bodies))
        inv_phi = pm.Exponential("inv_phi", 1.0)
        phi = pm.Deterministic("phi", 1.0 / inv_phi)
        log_mu = (np.log(E) + alpha + fam_term
                  + beta_renter * zr + u[bidx])
        if len(routes) > 1:
            gr = pm.Normal("gamma_route_raw", 0, 1.5, shape=len(routes) - 1)
            gamma = pm.Deterministic(
                "gamma_route", pm.math.concatenate([[0.0], gr]))
            log_mu = log_mu + gamma[ridx]
        if with_partial:
            b_part = pm.Normal("beta_partial", 0, 1.0)
            log_mu = log_mu + b_part * part
        mu = pm.Deterministic("mu", pm.math.exp(log_mu))
        pm.NegativeBinomial("y", mu=mu, alpha=phi, observed=y)
        idata = pm.sample(draws=DRAWS, tune=TUNE, chains=CHAINS,
                          target_accept=0.95, random_seed=SEED,
                          progressbar=False, idata_kwargs={"log_likelihood": False})
        pm.sample_posterior_predictive(
            idata, var_names=["y"], extend_inferencedata=True,
            random_seed=SEED, progressbar=False)

    names = ["alpha", "beta_renter", "sigma_b", "phi"] + \
            (["beta_partial"] if with_partial else [])
    names = [n for n in names if n in idata.posterior.data_vars]
    res = dict(label=label, n_cells=len(cells), n_bodies=len(bodies),
               families=fams, routes=routes, renter_mean=rmu, renter_sd=rsd,
               cells=[f"{c['route']}/{c['body']}/{c['family']}" for c in cells],
               params=summarise(idata, names), diagnostics=diag(idata))
    d = np.asarray(idata.posterior["beta_renter"]).ravel()
    rr = np.exp(d * 0.10 / rsd)
    res["rate_ratio_per_10pt_renter"] = dict(
        median=float(np.median(rr)),
        hdi90=[float(np.percentile(rr, 5)), float(np.percentile(rr, 95))],
        hdi95=[float(np.percentile(rr, 2.5)), float(np.percentile(rr, 97.5))],
        pr_gt_1=float((rr > 1).mean()))
    ppc = np.asarray(idata.posterior_predictive["y"]).reshape(-1, len(cells))
    lo, hi = np.percentile(ppc, 5, axis=0), np.percentile(ppc, 95, axis=0)
    res["ppc"] = dict(cells_inside_90=int(((y >= lo) & (y <= hi)).sum()),
                      cells=len(cells))
    rate = y / np.maximum(E, 1e-9)
    res["ols_unweighted"] = ols([[c["renter_share"] for c in cells]], rate)
    res["ols_hours_weighted"] = ols([[c["renter_share"] for c in cells]], rate, w=E)
    return res, idata


def forest(results, fn):
    labs, med, lo90, hi90, lo95, hi95 = [], [], [], [], [], []
    for r in results:
        p = r["params"].get("beta_renter")
        if not p:
            continue
        labs.append(f"{r['label']}  (n={r['n_cells']})")
        med.append(p["median"])
        lo90.append(p["hdi90"][0]); hi90.append(p["hdi90"][1])
        lo95.append(p["hdi95"][0]); hi95.append(p["hdi95"][1])
    yy = np.arange(len(labs))[::-1]
    fig, ax = plt.subplots(figsize=(8.4, 0.8 + 0.6 * len(labs)))
    ax.hlines(yy, lo95, hi95, color="#90cdf4", lw=3)
    ax.hlines(yy, lo90, hi90, color="#2b6cb0", lw=6)
    ax.plot(med, yy, "o", color="#1a365d", zorder=4)
    ax.axvline(0, color="#742a2a", lw=1, ls="--")
    ax.set_yticks(yy); ax.set_yticklabels(labs, fontsize=8)
    ax.set_xlabel("beta_renter — posterior per 1 sd of renter share\n"
                  "(log-odds for the binomial fits, log rate ratio for the NB fits)")
    ax.set_title("Renter share as a body-level predictor of meeting outcomes\n"
                 "90 % (dark) and 95 % (light) HDIs", fontsize=10)
    ax.grid(axis="x", alpha=0.25)
    fig.text(0.01, 0.01, STAMP + "  ·  every interval spans zero where it crosses the dashed line",
             fontsize=7, color="#742a2a")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(MOUT / fn, dpi=140)
    plt.close(fig)


def ppc_figure(res, fn):
    """Observed NIMBY count per cell against its replicated 90 % band
    (arviz 1.x dropped plot_ppc; this is the per-cell check that matters
    with 20 cells)."""
    cells = res["ppc"]["per_cell"]
    yy = np.arange(len(cells))[::-1]
    fig, ax = plt.subplots(figsize=(8.0, 0.7 + 0.34 * len(cells)))
    ax.hlines(yy, [c["rep_lo90"] for c in cells], [c["rep_hi90"] for c in cells],
              color="#90cdf4", lw=6, label="replicated 90 %")
    ax.plot([c["obs"] for c in cells], yy, "o", color="#1a365d",
            label="observed NIMBY turns", zorder=4)
    ax.set_yticks(yy)
    ax.set_yticklabels([f"{c['cell']} (N={c['N']})" for c in cells], fontsize=7)
    ax.set_xlabel("NIMBY count among public turns with a stance")
    inside = res["ppc"]["cells_inside_90"]
    ax.set_title(f"PPC — binomial NIMBY-count fit: {inside}/{len(cells)} cells "
                 f"inside the replicated 90 % band", fontsize=10)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(axis="x", alpha=0.25)
    fig.text(0.01, 0.01, STAMP, fontsize=7, color="#742a2a")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(MOUT / fn, dpi=140)
    plt.close(fig)


def main():
    MOUT.mkdir(parents=True, exist_ok=True)
    rows = [r for r in csv.DictReader(open(OUT / "outcomes_by_body_route.csv"))]
    for r in rows:
        for k in ("renter_share", "med_hh_income", "pop_density",
                  "meeting_hours"):
            r[k] = float(r[k]) if r[k] not in ("", None) else None
        for k in ("nimby_count", "yimby_count", "public_turns_numeric",
                  "dais_turns_numeric"):
            r[k] = int(r[k] or 0)

    stance_cells = [r for r in rows if r["public_turns_numeric"] >= MIN_N]
    print(f"model 1 cells (floor {MIN_N} public turns): {len(stance_cells)}, "
          f"{len(set(r['body'] for r in stance_cells))} bodies, "
          f"{sum(r['public_turns_numeric'] for r in stance_cells)} turns")

    results = []
    r1, i1 = fit_binomial(stance_cells, "NIMBY share ~ renter (route stratum)")
    results.append(r1)
    r2, _ = fit_binomial(stance_cells,
                         "NIMBY share ~ renter + log income + log density",
                         controls=True)
    results.append(r2)
    strict = [r for r in stance_cells if r["public_turns_numeric"] >= 10]
    if len(strict) >= 6:
        r3, _ = fit_binomial(strict, "NIMBY share ~ renter (floor 10 turns)")
        results.append(r3)
    # YIMBY count as the mirror outcome
    ycells = [dict(r, nimby_count=r["yimby_count"]) for r in stance_cells]
    r4, _ = fit_binomial(ycells, "YIMBY share ~ renter (route stratum)")
    results.append(r4)

    # ---- model 2 cells
    cov = {r["body"]: r for r in csv.DictReader(open(OUT / "covariates.csv"))}
    cells, cells_all = [], []
    for c in csv.DictReader(open(ROOT / "output" / "wave1" /
                                 "rates_by_body_route.csv")):
        if c["family"] not in ("density", "transit"):
            continue
        if c["body"] not in cov or not c["meeting_hours"]:
            continue
        if float(c["meeting_hours"]) <= 0:
            continue
        rec = dict(body=c["body"], route=c["route"], family=c["family"],
                   coverage=c["coverage"],
                   meeting_hours=float(c["meeting_hours"]),
                   episode_minutes=float(c["episode_minutes"] or 0),
                   renter_share=float(cov[c["body"]]["renter_share"]),
                   med_hh_income=float(cov[c["body"]]["med_hh_income"]),
                   pop_density=float(cov[c["body"]]["pop_density_per_sq_mi"]))
        cells_all.append(rec)
        if c["coverage"] == "complete":
            cells.append(rec)
    print(f"model 2 cells: {len(cells)} complete, {len(cells_all)} incl. partial; "
          f"{len(set(c['body'] for c in cells))} / "
          f"{len(set(c['body'] for c in cells_all))} bodies")

    n1, _ = fit_nb(cells, "topic minutes/h ~ renter (complete coverage)")
    results.append(n1)
    n2, _ = fit_nb(cells_all, "topic minutes/h ~ renter (+ partial, flagged)",
                   with_partial=True)
    results.append(n2)
    dens = [c for c in cells_all if c["family"] == "density"]
    n3, _ = fit_nb(dens, "density minutes/h ~ renter (all coverage)",
                   with_partial=True)
    results.append(n3)

    forest(results, "fig_beta_renter_forest.png")
    ppc_figure(r1, "fig_ppc_binomial.png")

    (MOUT / "model_results.json").write_text(json.dumps(
        dict(stamp=STAMP, seed=SEED, draws=DRAWS, tune=TUNE, chains=CHAINS,
             floor_public_turns=MIN_N, route_reference=ROUTE_REF,
             results=results), indent=2, default=str))

    with open(MOUT / "model_table.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "n_cells", "beta_renter_median", "hdi90_lo",
                    "hdi90_hi", "hdi95_lo", "hdi95_hi", "pr_gt_0",
                    "effect_per_10pt", "per10_lo90", "per10_hi90",
                    "max_r_hat", "min_ess", "divergences", "ppc_inside_90",
                    "ols_slope", "ols_lo95", "ols_hi95"])
        for r in results:
            p = r["params"]["beta_renter"]
            e = r.get("beta_per_10pt_renter") or r.get("rate_ratio_per_10pt_renter")
            o = r.get("ols_unweighted", {})
            w.writerow([r["label"], r["n_cells"], f"{p['median']:.4f}",
                        f"{p['hdi90'][0]:.4f}", f"{p['hdi90'][1]:.4f}",
                        f"{p['hdi95'][0]:.4f}", f"{p['hdi95'][1]:.4f}",
                        f"{p['pr_gt_0']:.3f}",
                        f"{e['median']:.4f}", f"{e['hdi90'][0]:.4f}",
                        f"{e['hdi90'][1]:.4f}",
                        f"{r['diagnostics']['max_r_hat']:.4f}",
                        f"{r['diagnostics']['min_ess_bulk']:.0f}",
                        r["diagnostics"]["divergences"],
                        f"{r['ppc']['cells_inside_90']}/{r['ppc']['cells']}",
                        f"{o.get('slope', float('nan')):.4f}",
                        f"{o.get('lo95', float('nan')):.4f}",
                        f"{o.get('hi95', float('nan')):.4f}"])

    for r in results:
        p = r["params"]["beta_renter"]
        e = r.get("beta_per_10pt_renter") or r.get("rate_ratio_per_10pt_renter")
        print(f"\n{r['label']}  n={r['n_cells']}")
        print(f"  beta_renter {p['median']:+.3f}  90% [{p['hdi90'][0]:+.3f},"
              f"{p['hdi90'][1]:+.3f}]  95% [{p['hdi95'][0]:+.3f},"
              f"{p['hdi95'][1]:+.3f}]  Pr(>0)={p['pr_gt_0']:.2f}")
        print(f"  per +10 pt renter: {e['median']:+.3f} "
              f"[{e['hdi90'][0]:+.3f},{e['hdi90'][1]:+.3f}]")
        print(f"  R-hat {r['diagnostics']['max_r_hat']:.4f}  ESS "
              f"{r['diagnostics']['min_ess_bulk']:.0f}  div "
              f"{r['diagnostics']['divergences']}  PPC "
              f"{r['ppc']['cells_inside_90']}/{r['ppc']['cells']}")
        o = r.get("ols_unweighted")
        if o and "lo95" in o:
            print(f"  OLS slope {o['slope']:+.4f} 95% [{o['lo95']:+.4f},"
                  f"{o['hi95']:+.4f}]")
    print(f"\n-> {MOUT}")


if __name__ == "__main__":
    main()
