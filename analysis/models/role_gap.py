#!/usr/bin/env python3
"""
role_gap.py -- estimand 2 (plan/SHOWCASE.md 1.2, ANALYSIS-transit.md 4.1): is
public comment more opposed / more hostile than the dais on the same item?
Cumulative-logit (proportional odds) model per turn:

  y_i in {1..K} ordered;  P(y_i <= k) = logistic(c_k - eta_i)
  eta_i = beta_role * public_i + beta_fam[f(i)] + beta_int[f(i)] * public_i
          + beta_len * (log words_i - mean) + beta_strat[s(i)] + a_body[b(i)] + a_ep[e(i)]

Outcomes: stance_median (-2..2, K = 5) and hostility_median (0..4, K = 5) from
the `turn_stance` table the stance pass writes (pipeline/stance.py, WP A). The
pass has NOT run as of 2026-10-17; until it does, load.standin_turns() builds
the same columns from VADER compound and hostility word-list hits, and every
output from that path says STAND-IN: VADER, not stance. The real-table path is
exercised by a synthetic test (analysis/test_pipeline.py, section 32).

FACET. A role contrast measured with two role instruments (episodes.roles_source
db / computed) is two measurements; fit() takes turns from ONE level and the
sweep runs it per level. The locate route enters as a stratum inside each fit,
stated in the footnote. Model statement and priors: README 2.

    .venv/bin/python analysis/models/role_gap.py --out output/analysis/models
"""
import argparse, sys, time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from analysis.models import load as L  # noqa: E402

CI = (5, 95)
ROLES = ("dais", "public")
OUTCOMES = {"stance": ("stance_median", (-2, -1, 0, 1, 2), "opposed (<= -1)"),
            "hostility": ("hostility_median", (0, 1, 2, 3, 4), "hostile (>= 2)")}
PRIORS = {
    "cutpoints": ("Normal(0, 3), ordered", "K-1 thresholds on the logit scale; sd 3 lets a category hold anything from <1 % to >95 % of turns"),
    "beta_role": ("Normal(0, 1.5)", "public-vs-dais shift in log-odds; 1.5 is an odds ratio of 4.5 at one sd, no direction assumed"),
    "beta_fam": ("Normal(0, 1.5), reference density", "family main effect on the same scale"),
    "beta_int": ("Normal(0, 1.0)", "role x family interaction, tighter than the main effects: the gap may differ by family, less than it exists"),
    "beta_len": ("Normal(0, 1.0) per log-word", "length control; the stand-in needs it (A1-RESULTS 1), the real pass keeps it as a check"),
    "beta_strat": ("Normal(0, 1.5), reference largest level", "locate-route stratum inside one role instrument"),
    "sigma_body": ("HalfNormal(1)", "between-body sd of the latent stance; 95 % under a 2-logit spread"),
    "sigma_ep": ("HalfNormal(1)", "between-episode sd within body: the item being discussed moves everyone in the room"),
}
MIN_TURNS = 60


def prepare(turns, outcome="stance"):
    col, levels, _ = OUTCOMES[outcome]
    d = turns[turns["role"].isin(ROLES) & turns[col].notna()].copy()
    d = d[d["words"] > 0]
    d["y"] = d[col].astype(int).map({v: i for i, v in enumerate(levels)})
    d = d[d["y"].notna()].copy()
    d["y"] = d["y"].astype(int)
    d["public"] = (d["role"] == "public").astype(int)
    d["log_words_c"] = np.log(d["words"]) - np.log(d["words"]).mean()
    return d.reset_index(drop=True)


def fit(turns, outcome="stance", stratum="method", draws=1000, tune=1000, chains=4, seed=20261017,
        target_accept=0.9, progress=False):
    """Cumulative-logit fit on turns from ONE role instrument. Returns dict."""
    import pymc as pm
    col, levels, _ = OUTCOMES[outcome]
    K = len(levels)
    d = prepare(turns, outcome)
    if len(d) < MIN_TURNS:
        raise ValueError(f"role_gap.fit: {len(d)} usable turns, under {MIN_TURNS}")
    if "role_source" in d.columns and d["role_source"].nunique() > 1:
        raise L.FacetError("role_gap.fit: turns span role instruments %s; fit per level (sweep)" % sorted(d["role_source"].unique()))
    fams = sorted(d["family"].unique(), key=lambda f: (f != "density", f))
    bodies = sorted(d["body"].unique()); eps = sorted(d["episode_id"].unique())
    strata = []
    if stratum in d.columns and d[stratum].nunique() > 1:
        strata = sorted(d[stratum].unique(), key=lambda s: (-(d[stratum] == s).sum(), s))
    fi = d["family"].map({f: i for i, f in enumerate(fams)}).to_numpy()
    bi = d["body"].map({b: i for i, b in enumerate(bodies)}).to_numpy()
    ei = d["episode_id"].map({e: i for i, e in enumerate(eps)}).to_numpy()
    si = d[stratum].map({s: i for i, s in enumerate(strata)}).to_numpy() if strata else None
    pub = d["public"].to_numpy(); lw = d["log_words_c"].to_numpy(); y = d["y"].to_numpy()
    coords = {"family": fams, "body": bodies, "episode": eps, "cut": list(range(K - 1))}
    if strata:
        coords["stratum"] = strata
    with pm.Model(coords=coords) as model:
        cut = pm.Normal("cutpoints", mu=np.linspace(-1.5, 1.5, K - 1), sigma=3.0, dims="cut",
                        transform=pm.distributions.transforms.ordered)
        b_role = pm.Normal("beta_role", 0.0, 1.5)
        b_fam_free = pm.Normal("beta_fam_free", 0.0, 1.5, shape=len(fams) - 1) if len(fams) > 1 else None
        b_int_free = pm.Normal("beta_int_free", 0.0, 1.0, shape=len(fams) - 1) if len(fams) > 1 else None
        if len(fams) > 1:
            b_fam = pm.Deterministic("beta_fam", pm.math.concatenate([[0.0], b_fam_free]), dims="family")
            b_int = pm.Deterministic("beta_int", pm.math.concatenate([[0.0], b_int_free]), dims="family")
        else:
            b_fam = pm.Deterministic("beta_fam", pm.math.zeros(1), dims="family")
            b_int = pm.Deterministic("beta_int", pm.math.zeros(1), dims="family")
        b_len = pm.Normal("beta_len", 0.0, 1.0)
        s_body = pm.HalfNormal("sigma_body", 1.0); z_body = pm.Normal("z_body", 0.0, 1.0, dims="body")
        a_body = pm.Deterministic("a_body", s_body * z_body, dims="body")
        s_ep = pm.HalfNormal("sigma_ep", 1.0); z_ep = pm.Normal("z_ep", 0.0, 1.0, dims="episode")
        a_ep = pm.Deterministic("a_ep", s_ep * z_ep, dims="episode")
        eta = (b_role * pub + b_fam[fi] + b_int[fi] * pub + b_len * lw + a_body[bi] + a_ep[ei])
        if strata:
            b_s_free = pm.Normal("beta_strat_free", 0.0, 1.5, shape=len(strata) - 1)
            b_s = pm.Deterministic("beta_strat", pm.math.concatenate([[0.0], b_s_free]), dims="stratum")
            eta = eta + b_s[si]
        pm.OrderedLogistic("y", eta=eta, cutpoints=cut, observed=y, compute_p=False)
        # the gap per family on the latent scale, and on the probability scale at
        # the mean body / episode / length, reference stratum
        gap = pm.Deterministic("gap", b_role + b_int, dims="family")
        pm.Deterministic("gap_or", pm.math.exp(gap), dims="family")
        t0 = time.time()
        idata = pm.sample(draws=draws, tune=tune, chains=chains, cores=min(chains, 4), random_seed=seed,
                          target_accept=target_accept, progressbar=progress)
        secs = time.time() - t0
        pm.sample_posterior_predictive(idata, var_names=["y"], extend_inferencedata=True, random_seed=seed, progressbar=progress)
    out = {"idata": idata, "data": d, "outcome": outcome, "levels": levels, "families": fams, "bodies": bodies,
           "strata": strata, "stratum": stratum, "seconds": secs, "n_turns": int(len(d)), "n_episodes": len(eps),
           "instrument": str(d["instrument"].iloc[0]) if "instrument" in d.columns else "unknown",
           "role_source": str(d["role_source"].iloc[0]) if "role_source" in d.columns else "unknown",
           "settings": {"draws": draws, "tune": tune, "chains": chains, "target_accept": target_accept, "seed": seed}}
    out["diagnostics"] = _diag(idata)
    return out


def _diag(idata):
    import arviz as az
    names = [v for v in idata.posterior.data_vars if v not in ("z_body", "z_ep", "a_ep", "a_body", "beta_fam_free", "beta_int_free", "beta_strat_free", "gap_or")]
    rhat = az.rhat(idata, var_names=names); ess = az.ess(idata, var_names=names)
    rh = float(max(float(np.nanmax(rhat[v].to_numpy())) for v in names))
    es = float(min(float(np.nanmin(ess[v].to_numpy())) for v in names))
    div = int(idata.sample_stats["diverging"].sum()) if "diverging" in idata.sample_stats else -1
    n = int(idata.posterior.sizes["chain"] * idata.posterior.sizes["draw"])
    return {"max_rhat": round(rh, 4), "min_ess_bulk": round(es, 0), "divergences": div, "draws_total": n,
            "ok": bool(rh < 1.01 and div == 0 and es >= 400)}


def _flat(idata, name):
    return idata.posterior[name].stack(sample=("chain", "draw")).transpose("sample", ...).to_numpy()


def _q(x, axis=0):
    lo, hi = np.percentile(x, CI, axis=axis)
    return np.median(x, axis=axis), lo, hi


def _logistic(x):
    return 1.0 / (1.0 + np.exp(-x))


def gap_table(fit_):
    """Per family: latent gap, odds ratio, and the probability-scale gap in the
    'opposed' (stance <= -1) or 'hostile' (>= 2) share at the mean body/episode."""
    idata = fit_["idata"]; fams = fit_["families"]; levels = fit_["levels"]
    cut = _flat(idata, "cutpoints"); gap = _flat(idata, "gap"); bf = _flat(idata, "beta_fam")
    rows = []
    if fit_["outcome"] == "stance":
        k = levels.index(-1)              # P(y <= -1) = P(z <= k)
        tail = lambda eta: _logistic(cut[:, k] - eta)
    else:
        k = levels.index(2)               # P(y >= 2) = 1 - P(z <= 1)
        tail = lambda eta: 1.0 - _logistic(cut[:, k - 1] - eta)
    for i, f in enumerate(fams):
        eta_d = bf[:, i]; eta_p = bf[:, i] + gap[:, i]
        pd_, pp = tail(eta_d), tail(eta_p)
        m, l, h = _q(gap[:, i]); om, ol, oh = _q(np.exp(gap[:, i]))
        dm, dl, dh = _q(pp - pd_)
        n_f = fit_["data"][fit_["data"]["family"] == f]
        rows.append({"family": f, "gap_latent_median": m, "gap_lo90": l, "gap_hi90": h,
                     "odds_ratio_median": om, "or_lo90": ol, "or_hi90": oh,
                     "pr_gap_gt_0": float((gap[:, i] > 0).mean()),
                     "p_tail_dais": float(np.median(pd_)), "p_tail_public": float(np.median(pp)),
                     "p_tail_diff_median": dm, "p_tail_diff_lo90": dl, "p_tail_diff_hi90": dh,
                     "tail": OUTCOMES[fit_["outcome"]][2],
                     "n_turns": int(len(n_f)), "n_public": int(n_f["public"].sum()),
                     "instrument": fit_["instrument"], "role_source": fit_["role_source"]})
    # the pooled-over-family gap: average of family gaps weighted by turns
    w = np.array([(fit_["data"]["family"] == f).sum() for f in fams], float); w /= w.sum()
    g_all = gap @ w; m, l, h = _q(g_all)
    rows.append({"family": "all (turn-weighted)", "gap_latent_median": m, "gap_lo90": l, "gap_hi90": h,
                 "odds_ratio_median": float(np.median(np.exp(g_all))), "or_lo90": float(np.exp(l)), "or_hi90": float(np.exp(h)),
                 "pr_gap_gt_0": float((g_all > 0).mean()), "p_tail_dais": np.nan, "p_tail_public": np.nan,
                 "p_tail_diff_median": np.nan, "p_tail_diff_lo90": np.nan, "p_tail_diff_hi90": np.nan,
                 "tail": OUTCOMES[fit_["outcome"]][2], "n_turns": int(len(fit_["data"])),
                 "n_public": int(fit_["data"]["public"].sum()), "instrument": fit_["instrument"], "role_source": fit_["role_source"]})
    return pd.DataFrame(rows)


def coef_table(fit_):
    idata = fit_["idata"]; rows = []
    for name in ("beta_role", "beta_len", "sigma_body", "sigma_ep"):
        x = _flat(idata, name); m, l, h = _q(x)
        rows.append({"parameter": name, "level": "", "median": m, "lo90": l, "hi90": h})
    for name, labels in (("beta_fam", fit_["families"]), ("beta_int", fit_["families"]),
                         ("beta_strat", fit_["strata"]), ("cutpoints", [f"c{k}" for k in range(len(fit_["levels"]) - 1)])):
        if name not in idata.posterior:
            continue
        x = _flat(idata, name); m, l, h = _q(x)
        for j, lab in enumerate(labels):
            rows.append({"parameter": name, "level": lab, "median": m[j], "lo90": l[j], "hi90": h[j]})
    return pd.DataFrame(rows)


def body_table(fit_):
    x = _flat(fit_["idata"], "a_body"); m, l, h = _q(x)
    d = fit_["data"]
    return pd.DataFrame({"body": fit_["bodies"], "intercept_median": m, "lo90": l, "hi90": h,
                         "n_turns": [int((d["body"] == b).sum()) for b in fit_["bodies"]],
                         "n_public": [int(d[d["body"] == b]["public"].sum()) for b in fit_["bodies"]]}
                        ).sort_values("intercept_median").reset_index(drop=True)


def ppc_table(fit_):
    y = fit_["data"]["y"].to_numpy()
    rep = fit_["idata"].posterior_predictive["y"].stack(sample=("chain", "draw")).transpose("sample", ...).to_numpy()
    rows = []
    for role in ROLES:
        mask = (fit_["data"]["role"] == role).to_numpy()
        for k, lev in enumerate(fit_["levels"]):
            obs = (y[mask] == k).mean(); r = (rep[:, mask] == k).mean(axis=1)
            lo, hi = np.percentile(r, CI)
            rows.append({"role": role, "level": lev, "observed_share": float(obs), "rep_median": float(np.median(r)),
                         "rep_lo90": float(lo), "rep_hi90": float(hi), "n": int(mask.sum())})
    return pd.DataFrame(rows)


def summarise(fit_):
    return {"gap": gap_table(fit_), "coef": coef_table(fit_), "bodies": body_table(fit_), "ppc": ppc_table(fit_)}


def sweep(turns, facet="role_source", outcomes=("stance", "hostility"), min_turns=MIN_TURNS, **kw):
    """One fit per facet level per outcome. Levels under min_turns are reported, not fitted."""
    fits, skipped = {}, {}
    levels = sorted(turns[facet].unique()) if facet in turns.columns else ["all"]
    for lev in levels:
        t = turns[turns[facet] == lev] if facet in turns.columns else turns
        for oc in outcomes:
            n = len(prepare(t, oc))
            if n < min_turns:
                skipped[(lev, oc)] = {"n_turns": int(n), "reason": f"fewer than {min_turns} usable turns"}
                continue
            fits[(lev, oc)] = fit(t, outcome=oc, **kw)
    return fits, skipped


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(ROOT / "output" / "analysis" / "models"))
    ap.add_argument("--corpus", default=str(L.DEFAULT_CORPUS))
    ap.add_argument("--state", default=None)
    ap.add_argument("--draws", type=int, default=1000)
    ap.add_argument("--tune", type=int, default=1000)
    ap.add_argument("--chains", type=int, default=4)
    a = ap.parse_args(argv)
    from analysis.models import run
    return run.role_gap_stage(Path(a.out), corpus=a.corpus, state=a.state, draws=a.draws, tune=a.tune, chains=a.chains)


if __name__ == "__main__":
    sys.exit(main())
