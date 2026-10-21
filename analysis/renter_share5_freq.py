#!/usr/bin/env python3
"""
renter_share5_freq.py -- the FREQUENTIST counterpart to the Bayesian
renter-share/role regressions (runs 2 and 3), built to the standard set by
plan/REGRESSION-REVIEW-2026-10-19.md findings 2, 3, 4, 6, 7 and 10.

PRIMARY OUTCOME (finding 2). NIMBY out of NIMBY+YIMBY -- the DIRECTIONAL
denominator -- with `neutral` dropped and staff clusters dropped. The published
runs put `neutral` in the denominator, where it is ~60 % of the mass and a dais
phenomenon, so the published delta = +2.11 is mostly the neutral class plus
staff reports. SECONDARY: the published denominator (clusters with a stance,
staff kept), refitted unchanged, so the two are comparable.

FITS
  1  GEE binomial logit, exchangeable working correlation clustered by BODY,
     robust sandwich SEs -> population-averaged estimates + 95 % Wald CIs.
     Plus an independence working correlation as a check.
  2  Binomial GLMM with a body random intercept, by MARGINAL MAXIMUM
     LIKELIHOOD (Gauss-Hermite quadrature, custom; R/lme4 and glmmTMB are not
     installed on this Mac and statsmodels' BinomialBayesMixedGLM is a
     variational BAYES fit, not a frequentist GLMM). Wald + profile-likelihood
     CIs, the random-intercept SD, and LRTs for role.
  3  Beta-binomial GLM (custom MLE) as the overdispersion-aware fixed-effects
     fit; reports the intra-row correlation rho.
  4  Nested-structure check (finding 4): turn-level logistic, NIMBY turn vs
     YIMBY turn, cluster-robust by EPISODE, then MEETING, then BODY.
  5  Leave-one-body-out (finding 6), grouped 5-fold by body with out-of-sample
     log-loss (finding 7), Freedman-Lane permutation for renter share
     (finding 10).

Design, matched to run 3 so the coefficients are comparable: eight kept
covariates (output/renter_share_2/screen.json) standardised over the fitted
rows, role (dais=0, public=1), role x z_renter, route dummies (classifier is
the reference stratum, route is never pooled).

Writes output/renter_share_5/freq/. Read-only on the DB and on runs 2-4; no
network, no spend, no API, nothing committed.

    nohup .venv/bin/python analysis/renter_share5_freq.py \
        > output/renter_share_5/freq.log 2>&1 &
"""
import collections
import csv
import os
import json
import pathlib
import sqlite3
import time

import numpy as np
import statsmodels.api as sm
from scipy import optimize, stats
from scipy.special import expit, gammaln, logsumexp

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT2 = ROOT / "output" / "renter_share_2"
OUT3 = ROOT / "output" / "renter_share_3"
OUT = ROOT / "output" / "renter_share_5" / "freq"
CLUSTERS = ROOT / "output" / "stance_s1" / "nimby_view" / "speaker_clusters_roleaudit.csv"
DB = pathlib.Path.home() / ".local" / "share" / "ytpt" / "state.sqlite"

MIN_N = 5
REF_ROUTE = "classifier"
N_PERM = int(os.environ.get("FREQ_NPERM", "5000"))
GH_NODES = 24
SEED = 20261019
rng = np.random.default_rng(SEED)

KEPT = json.loads((OUT2 / "screen.json").read_text())["kept"]
LOGCOLS = {"log_income": "med_hh_income", "log_density": "pop_density_per_sq_mi",
           "log_home_value": "med_home_value", "log_gross_rent": "med_gross_rent",
           "log_population": "population"}


def log(*a):
    print(*a, flush=True)


# ----------------------------------------------------------------- inputs

def covariates():
    out = {}
    for r in csv.DictReader(open(OUT2 / "covariates_wide.csv")):
        d = {k: float(r[k]) for k in (
            "renter_share", "med_hh_income", "population", "med_home_value",
            "med_gross_rent", "rent_burden_30", "built_2010_plus",
            "multiunit_5plus", "ba_plus", "median_age", "nhwhite_share",
            "transit_commute", "vacancy_rate", "pop_density_per_sq_mi")}
        for lk, raw in LOGCOLS.items():
            d[lk] = float(np.log(d[raw]))
        out[r["body"]] = d
    return out


def staff_clusters(con):
    """(episode_id, cluster) whose turns' modal self_id is 'staff'."""
    per = collections.defaultdict(collections.Counter)
    for ep, cl, sid in con.execute(
            "select episode_id, cluster, self_id from turn_stance"):
        per[(ep, cl)][sid or "unknown"] += 1
    staff = set()
    for key, c in per.items():
        known = {k: v for k, v in c.items() if k != "unknown"}
        if known and max(known, key=known.get) == "staff":
            staff.add(key)
    return staff


def clusters(staff):
    rows = []
    for r in csv.DictReader(open(CLUSTERS)):
        if r["role"] not in ("dais", "public"):
            continue
        rows.append(dict(episode_id=r["episode_id"], cluster=r["cluster"],
                         role=r["role"], body=r["body"], route=r["method"],
                         label=r["label"], n_numeric=int(r["n_numeric"]),
                         staff=int((r["episode_id"], r["cluster"]) in staff)))
    return rows


def build(cl, cov, directional, drop_staff):
    """body x route x role rows under one denominator rule."""
    sel = [c for c in cl if not (drop_staff and c["staff"])]
    agg = collections.defaultdict(lambda: collections.Counter())
    eps = collections.defaultdict(set)
    for c in sel:
        key = (c["route"], c["body"], c["role"])
        agg[key][c["label"]] += 1
        eps[key].add(c["episode_id"])
    rows = []
    for (route, body, role), c in sorted(agg.items()):
        if body not in cov:
            continue
        n = (c["NIMBY"] + c["YIMBY"]) if directional else (
            c["NIMBY"] + c["YIMBY"] + c["neutral"])
        if n < MIN_N:
            continue
        rows.append(dict(route=route, body=body, role=role,
                         role_code=0 if role == "dais" else 1,
                         y=c["NIMBY"], N=n, episodes=len(eps[(route, body, role)]),
                         **cov[body]))
    return rows


def design(rows, cols=None):
    """X, names. Covariates standardised over the fitted rows (as in run 3)."""
    Z = np.array([[r[k] for k in KEPT] for r in rows], float)
    mu, sd = Z.mean(0), Z.std(0, ddof=0)
    sd[sd == 0] = 1.0
    Z = (Z - mu) / sd
    role = np.array([r["role_code"] for r in rows], float)
    zren = Z[:, KEPT.index("renter_share")]
    routes = sorted({r["route"] for r in rows} - {REF_ROUTE})
    parts = [np.ones(len(rows)), role, zren, role * zren]
    names = ["const", "role", "renter_share", "role_x_renter"]
    for j, k in enumerate(KEPT):
        if k == "renter_share":
            continue
        parts.append(Z[:, j])
        names.append(k)
    for rt in routes:
        parts.append(np.array([1.0 if r["route"] == rt else 0.0 for r in rows]))
        names.append(f"route[{rt}]")
    X = np.column_stack(parts)
    keep = [names.index(c) for c in cols] if cols else list(range(len(names)))
    return X[:, keep], [names[i] for i in keep], dict(
        cov_mean=dict(zip(KEPT, mu.round(6).tolist())),
        cov_sd=dict(zip(KEPT, sd.round(6).tolist())))


def expand(rows, X):
    """Bernoulli-level expansion: one trial per cluster, grouped by body."""
    xs, ys, gs = [], [], []
    for i, r in enumerate(rows):
        for _ in range(r["y"]):
            xs.append(X[i]); ys.append(1.0); gs.append(r["body"])
        for _ in range(r["N"] - r["y"]):
            xs.append(X[i]); ys.append(0.0); gs.append(r["body"])
    return np.array(xs), np.array(ys), np.array(gs)


# ----------------------------------------------------------------- fits

def gee(rows, X, names, struct="exchangeable"):
    Xb, yb, gb = expand(rows, X)
    cs = (sm.cov_struct.Exchangeable() if struct == "exchangeable"
          else sm.cov_struct.Independence())
    m = sm.GEE(yb, Xb, groups=gb, family=sm.families.Binomial(), cov_struct=cs)
    res = m.fit(maxiter=120)
    ci = res.conf_int()
    out = {}
    for j, nm in enumerate(names):
        out[nm] = dict(est=float(res.params[j]), se=float(res.bse[j]),
                       lo=float(ci[j, 0]), hi=float(ci[j, 1]),
                       z=float(res.params[j] / res.bse[j]),
                       p=float(2 * stats.norm.sf(abs(res.params[j] / res.bse[j]))))
    alpha = None
    try:
        alpha = float(np.atleast_1d(res.cov_struct.dep_params)[0])
    except Exception:
        pass
    return out, dict(n_trials=int(len(yb)), n_bodies=int(len(set(gb))),
                     n_rows=len(rows), alpha=alpha, struct=struct), res


def loglik_binom(y, N, eta):
    p = np.clip(expit(eta), 1e-12, 1 - 1e-12)
    return (gammaln(N + 1) - gammaln(y + 1) - gammaln(N - y + 1)
            + y * np.log(p) + (N - y) * np.log(1 - p))


class GLMM:
    """Binomial GLMM, body random intercept, Gauss-Hermite marginal ML."""

    def __init__(self, rows, X):
        self.X, self.y = X, np.array([r["y"] for r in rows], float)
        self.N = np.array([r["N"] for r in rows], float)
        bodies = sorted({r["body"] for r in rows})
        self.groups = [np.array([i for i, r in enumerate(rows)
                                 if r["body"] == b]) for b in bodies]
        x, w = np.polynomial.hermite_e.hermegauss(GH_NODES)
        # hermegauss is for exp(-x^2/2); nodes are already on the N(0,1) scale
        self.nodes, self.lw = x, np.log(w) - 0.5 * np.log(2 * np.pi)

    def nll(self, theta):
        beta, sig = theta[:-1], np.exp(theta[-1])
        eta0 = self.X @ beta
        total = 0.0
        for idx in self.groups:
            e = eta0[idx][:, None] + sig * self.nodes[None, :]
            ll = loglik_binom(self.y[idx][:, None], self.N[idx][:, None], e).sum(0)
            total += logsumexp(ll + self.lw)
        return -total

    def fit(self, start=None):
        k = self.X.shape[1]
        if start is None:
            g = sm.GLM(np.column_stack([self.y, self.N - self.y]), self.X,
                       family=sm.families.Binomial()).fit()
            start = np.append(np.asarray(g.params, float), np.log(0.3))
        r = optimize.minimize(self.nll, start, method="L-BFGS-B",
                              options=dict(maxiter=4000, ftol=1e-13, gtol=1e-8))
        r = optimize.minimize(self.nll, r.x, method="Nelder-Mead",
                              options=dict(maxiter=3000, xatol=1e-6, fatol=1e-8))
        r = optimize.minimize(self.nll, r.x, method="L-BFGS-B",
                              options=dict(maxiter=4000, ftol=1e-13, gtol=1e-8))
        self.theta, self.ll = r.x, -r.fun
        return self

    def sigma_profile(self, grid):
        """loglik maximised over beta at fixed sigma -- boundary diagnostic."""
        out = {}
        j = len(self.theta) - 1
        for s in grid:
            val = np.log(max(s, 1e-8))
            out[float(s)] = round(-self.nll_fixed(j, val), 4)
        return out

    def nll_fixed(self, j, val):
        """profile: maximise over everything but coefficient j."""
        free = [i for i in range(len(self.theta)) if i != j]

        def f(z):
            t = self.theta.copy()
            t[j] = val
            t[free] = z
            return self.nll(t)
        r = optimize.minimize(f, self.theta[free], method="L-BFGS-B",
                              options=dict(maxiter=3000, ftol=1e-13, gtol=1e-8))
        return float(r.fun)

    def hess_se(self):
        t, n = self.theta, len(self.theta)
        h = np.maximum(1e-4, 1e-4 * np.abs(t))
        H = np.zeros((n, n))
        f0 = self.nll(t)
        for i in range(n):
            for j in range(i, n):
                tpp, tpm, tmp, tmm = t.copy(), t.copy(), t.copy(), t.copy()
                tpp[i] += h[i]; tpp[j] += h[j]
                tpm[i] += h[i]; tpm[j] -= h[j]
                tmp[i] -= h[i]; tmp[j] += h[j]
                tmm[i] -= h[i]; tmm[j] -= h[j]
                if i == j:
                    H[i, i] = (self.nll(tpp) - 2 * f0 + self.nll(tmm)) / (4 * h[i] ** 2)
                else:
                    H[i, j] = H[j, i] = ((self.nll(tpp) - self.nll(tpm)
                                          - self.nll(tmp) + self.nll(tmm))
                                         / (4 * h[i] * h[j]))
        try:
            return np.sqrt(np.abs(np.diag(np.linalg.inv(H))))
        except np.linalg.LinAlgError:
            return np.full(n, np.nan)


class BetaBinom:
    """Beta-binomial GLM: logit mean, scalar precision; custom MLE."""

    def __init__(self, rows, X):
        self.X = X
        self.y = np.array([r["y"] for r in rows], float)
        self.N = np.array([r["N"] for r in rows], float)

    def nll(self, theta):
        beta, phi = theta[:-1], np.exp(theta[-1])
        p = np.clip(expit(self.X @ beta), 1e-9, 1 - 1e-9)
        a, b = p * phi, (1 - p) * phi
        ll = (gammaln(self.N + 1) - gammaln(self.y + 1) - gammaln(self.N - self.y + 1)
              + gammaln(self.y + a) + gammaln(self.N - self.y + b) - gammaln(self.N + phi)
              + gammaln(phi) - gammaln(a) - gammaln(b))
        return -ll.sum()

    def fit(self):
        g = sm.GLM(np.column_stack([self.y, self.N - self.y]), self.X,
                   family=sm.families.Binomial()).fit()
        start = np.append(np.asarray(g.params, float), np.log(20.0))
        r = optimize.minimize(self.nll, start, method="L-BFGS-B",
                              options=dict(maxiter=4000, ftol=1e-13, gtol=1e-8))
        r = optimize.minimize(self.nll, r.x, method="Nelder-Mead",
                              options=dict(maxiter=4000, xatol=1e-6, fatol=1e-8))
        r = optimize.minimize(self.nll, r.x, method="L-BFGS-B",
                              options=dict(maxiter=4000, ftol=1e-13, gtol=1e-8))
        self.theta, self.ll = r.x, -r.fun
        return self

    def se(self):
        t = self.theta
        h = np.maximum(1e-5, 1e-4 * np.abs(t))
        n = len(t)
        H = np.zeros((n, n))
        f0 = self.nll(t)
        for i in range(n):
            for j in range(i, n):
                tpp, tpm, tmp, tmm = t.copy(), t.copy(), t.copy(), t.copy()
                tpp[i] += h[i]; tpp[j] += h[j]
                tpm[i] += h[i]; tpm[j] -= h[j]
                tmp[i] -= h[i]; tmp[j] += h[j]
                tmm[i] -= h[i]; tmm[j] -= h[j]
                if i == j:
                    H[i, i] = (self.nll(tpp) - 2 * f0 + self.nll(tmm)) / (4 * h[i] ** 2)
                else:
                    H[i, j] = H[j, i] = ((self.nll(tpp) - self.nll(tpm)
                                          - self.nll(tmp) + self.nll(tmm))
                                         / (4 * h[i] * h[j]))
        try:
            return np.sqrt(np.abs(np.diag(np.linalg.inv(H))))
        except np.linalg.LinAlgError:
            return np.full(n, np.nan)


# ----------------------------------------------------------------- turn level

def turn_level(con, staff, cov):
    role_of, route_of = {}, {}
    for r in csv.DictReader(open(CLUSTERS)):
        role_of[(r["episode_id"], r["cluster"])] = r["role"]
        route_of[(r["episode_id"], r["cluster"])] = r["method"]
    meeting = dict(con.execute("select episode_id, meeting_id from episodes"))
    rows = []
    q = ("select episode_id, cluster, body, stance_median from turn_stance "
         "where stance_median is not null")
    for ep, cl, body, sm_ in con.execute(q):
        key = (ep, cl)
        if key in staff or key not in role_of or body not in cov:
            continue
        if role_of[key] not in ("dais", "public") or abs(sm_) < 0.5:
            continue
        rows.append(dict(y=1.0 if sm_ <= -0.5 else 0.0, body=body,
                         episode_id=ep, meeting_id=meeting.get(ep, ep),
                         route=route_of[key], role=role_of[key],
                         role_code=0 if role_of[key] == "dais" else 1,
                         renter_share=cov[body]["renter_share"]))
    return rows


def turn_fits(rows):
    y = np.array([r["y"] for r in rows])
    role = np.array([r["role_code"] for r in rows], float)
    ren = np.array([r["renter_share"] for r in rows], float)
    zren = (ren - ren.mean()) / ren.std(ddof=0)
    routes = sorted({r["route"] for r in rows} - {REF_ROUTE})
    parts, names = [np.ones(len(rows)), role, zren], ["const", "role", "renter_share"]
    for rt in routes:
        parts.append(np.array([1.0 if r["route"] == rt else 0.0 for r in rows]))
        names.append(f"route[{rt}]")
    X = np.column_stack(parts)
    out = {}
    levels = [("naive (iid)", None), ("episode", "episode_id"),
              ("meeting", "meeting_id"), ("body", "body")]
    for lab, key in levels:
        if key is None:
            res = sm.Logit(y, X).fit(disp=0)
        else:
            g = np.array([r[key] for r in rows])
            res = sm.Logit(y, X).fit(disp=0, cov_type="cluster",
                                     cov_kwds=dict(groups=g, use_correction=True))
        out[lab] = dict(
            n_groups=(len({r[key] for r in rows}) if key else len(rows)),
            **{nm: dict(est=float(res.params[j]), se=float(res.bse[j]),
                        p=float(res.pvalues[j]))
               for j, nm in enumerate(names) if nm in ("role", "renter_share")})
    return out, dict(n_turns=len(rows), n_nimby=int(y.sum()),
                     n_episodes=len({r["episode_id"] for r in rows}),
                     n_meetings=len({r["meeting_id"] for r in rows}),
                     n_bodies=len({r["body"] for r in rows}))


# ----------------------------------------------------------------- 5: checks

def lobo(rows, names_want):
    bodies = sorted({r["body"] for r in rows})
    base_X, base_names, _ = design(rows)
    full, _, _ = gee(rows, base_X, base_names)
    out = []
    for b in bodies:
        sub = [r for r in rows if r["body"] != b]
        try:
            X, nm, _ = design(sub)
            est, _, _ = gee(sub, X, nm)
        except Exception as e:                                   # pragma: no cover
            log(f"  LOBO {b}: {e}")
            continue
        out.append(dict(dropped=b, n_rows=len(sub),
                        **{k: round(est[k]["est"], 4) for k in names_want
                           if k in est}))
    return full, out


def kfold(rows, k=5):
    bodies = sorted({r["body"] for r in rows})
    perm = rng.permutation(len(bodies))
    fold = {bodies[perm[i]]: i % k for i in range(len(bodies))}
    sets = {
        "intercept_only": ["const"],
        "route_only": ["const"],
        "+role": ["const", "role"],
        "+role+renter": ["const", "role", "renter_share", "role_x_renter"],
        "+full": None,
    }
    out = {}
    for lab, cols in sets.items():
        tot_ll, tot_n, unseen, skipped = 0.0, 0, 0, 0
        for f in range(k):
            tr = [r for r in rows if fold[r["body"]] != f]
            te = [r for r in rows if fold[r["body"]] == f]
            if not te or not tr:
                continue
            _, nmfull, _ = design(tr)
            routes_tr = [n for n in nmfull if n.startswith("route[")]
            want = (nmfull if cols is None
                    else [n for n in cols if n in nmfull]
                    + ([] if lab == "intercept_only" else routes_tr))
            unseen += sum(r["N"] for r in te
                          if lab != "intercept_only" and r["route"] != REF_ROUTE
                          and f"route[{r['route']}]" not in routes_tr)
            Xtr, nmtr, _ = design(tr, want)
            # test rows standardised on the TRAINING rows' moments
            Xte = design_on(te, tr, nmtr)
            # fold fits are binomial GLMs: the working correlation changes
            # SEs, not the fitted mean, and IRLS does not diverge here.
            ytr = np.array([[r["y"], r["N"] - r["y"]] for r in tr], float)
            try:
                gl = sm.GLM(ytr, Xtr, family=sm.families.Binomial()).fit(maxiter=200)
                par = np.asarray(gl.params, float)
                if not np.all(np.isfinite(par)):
                    raise ValueError("non-finite IRLS params")
                eta = np.clip(Xte @ par, -15.0, 15.0)
            except Exception as e:                               # pragma: no cover
                log(f"  kfold {lab} f{f}: {e}")
                skipped += 1
                continue
            p = np.clip(expit(eta), 1e-9, 1 - 1e-9)
            yv = np.array([r["y"] for r in te], float)
            Nv = np.array([r["N"] for r in te], float)
            tot_ll += float((yv * np.log(p) + (Nv - yv) * np.log(1 - p)).sum())
            tot_n += int(Nv.sum())
        out[lab] = dict(trials=tot_n, total_logdens=round(tot_ll, 2),
                        logloss=round(-tot_ll / tot_n, 4) if tot_n else None,
                        trials_route_unseen_in_train=int(unseen),
                        folds_skipped=int(skipped))
    return out, {b: fold[b] for b in bodies}


def design_on(test_rows, train_rows, names):
    """Design for test_rows using train_rows' standardisation moments."""
    Ztr = np.array([[r[k] for k in KEPT] for r in train_rows], float)
    mu, sd = Ztr.mean(0), Ztr.std(0, ddof=0)
    sd[sd == 0] = 1.0
    Z = (np.array([[r[k] for k in KEPT] for r in test_rows], float) - mu) / sd
    role = np.array([r["role_code"] for r in test_rows], float)
    zren = Z[:, KEPT.index("renter_share")]
    cols = {"const": np.ones(len(test_rows)), "role": role,
            "renter_share": zren, "role_x_renter": role * zren}
    for j, k in enumerate(KEPT):
        if k != "renter_share":
            cols[k] = Z[:, j]
    for nm in names:
        if nm.startswith("route["):
            rt = nm[6:-1]
            cols[nm] = np.array([1.0 if r["route"] == rt else 0.0
                                 for r in test_rows])
    return np.column_stack([cols[nm] for nm in names])


def freedman_lane(rows, n_perm=N_PERM):
    """Permute the renter-share residual (residualised on the other covariates
    and the design) across BODIES -- a valid null for a partial coefficient
    (finding 10)."""
    X, names, _ = design(rows)
    jr, ji = names.index("renter_share"), names.index("role_x_renter")
    other = [i for i in range(len(names)) if i not in (jr, ji)]
    # residualise z_renter on the rest of the design, at body level
    bodies = sorted({r["body"] for r in rows})
    bidx = {b: i for i, b in enumerate(bodies)}
    rowb = np.array([bidx[r["body"]] for r in rows])
    Xo = X[:, other]
    beta, *_ = np.linalg.lstsq(Xo, X[:, jr], rcond=None)
    resid = X[:, jr] - Xo @ beta
    fitted = Xo @ beta
    # one residual per body (constant within body up to the role/route terms):
    bres = np.array([resid[rowb == i].mean() for i in range(len(bodies))])
    obs, _, _ = gee(rows, X, names, struct="independence")
    t_obs = abs(obs["renter_share"]["z"])
    hits, zs = 0, []
    t0 = time.time()
    for p in range(n_perm):
        pr = bres[rng.permutation(len(bodies))]
        Xp = X.copy()
        Xp[:, jr] = fitted + pr[rowb]
        Xp[:, ji] = Xp[:, names.index("role")] * Xp[:, jr]
        try:
            est, _, _ = gee(rows, Xp, names, struct="independence")
            z = abs(est["renter_share"]["z"])
        except Exception:
            continue
        zs.append(z)
        hits += int(z >= t_obs)
        if p == 49:
            log(f"  FL: 50 perms in {time.time() - t0:.1f}s")
    n_ok = len(zs)
    return dict(t_obs=round(t_obs, 4), n_perm=n_ok,
                p_value=round((hits + 1) / (n_ok + 1), 4),
                perm_z_p95=round(float(np.percentile(zs, 95)), 3) if zs else None,
                note=("permutation refits use the independence working "
                      "correlation with by-body robust SEs for speed; the "
                      "observed statistic is recomputed the same way below"))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    cov = covariates()
    staff = staff_clusters(con)
    cl = clusters(staff)
    log(f"clusters {len(cl)}  staff-modal clusters in DB {len(staff)}  "
        f"staff among role clusters {sum(c['staff'] for c in cl)} "
        f"(dais {sum(c['staff'] for c in cl if c['role'] == 'dais')})")

    prim = build(cl, cov, directional=True, drop_staff=True)
    sec = build(cl, cov, directional=False, drop_staff=False)
    for lab, rows in (("primary (directional, no staff)", prim),
                      ("secondary (published denominator)", sec)):
        log(f"{lab}: rows {len(rows)} bodies {len({r['body'] for r in rows})} "
            f"trials {sum(r['N'] for r in rows)} nimby {sum(r['y'] for r in rows)}")
    with open(OUT / "rows_primary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["route", "body", "role", "role_code",
                                           "y", "N", "episodes"],
                           extrasaction="ignore")
        w.writeheader()
        for r in prim:
            w.writerow(r)

    res = dict(generated=time.strftime("%Y-%m-%dT%H:%M:%S"), seed=SEED,
               kept_covariates=KEPT, min_n=MIN_N, ref_route=REF_ROUTE,
               r_available=False, gh_nodes=GH_NODES,
               counts=dict(
                   staff_clusters=len(staff),
                   primary=dict(rows=len(prim),
                                bodies=len({r["body"] for r in prim}),
                                trials=sum(r["N"] for r in prim),
                                nimby=sum(r["y"] for r in prim)),
                   secondary=dict(rows=len(sec),
                                  bodies=len({r["body"] for r in sec}),
                                  trials=sum(r["N"] for r in sec),
                                  nimby=sum(r["y"] for r in sec))))

    for tag, rows in (("primary", prim), ("secondary", sec)):
        X, names, meta = design(rows)
        blk = dict(design=meta, names=names)
        for struct in ("exchangeable", "independence"):
            est, info, _ = gee(rows, X, names, struct)
            blk[f"gee_{struct}"] = dict(est=est, info=info)
            log(f"[{tag}] GEE {struct}: role {est['role']['est']:+.3f} "
                f"({est['role']['lo']:+.3f},{est['role']['hi']:+.3f})  "
                f"renter {est['renter_share']['est']:+.3f} "
                f"({est['renter_share']['lo']:+.3f},{est['renter_share']['hi']:+.3f})"
                f"  alpha={info['alpha']}")

        g = GLMM(rows, X).fit()
        se = g.hess_se()
        sig = float(np.exp(g.theta[-1]))
        blk["glmm"] = dict(
            loglik=round(g.ll, 4), sigma_body=round(sig, 4),
            est={nm: dict(est=float(g.theta[j]), se=float(se[j]),
                          lo=float(g.theta[j] - 1.96 * se[j]),
                          hi=float(g.theta[j] + 1.96 * se[j]))
                 for j, nm in enumerate(names)})
        log(f"[{tag}] GLMM: role {g.theta[names.index('role')]:+.3f} "
            f"sigma_body {sig:.3f} loglik {g.ll:.3f}")

        # profile CI + LRTs for role
        jrole = names.index("role")
        crit = stats.chi2.ppf(0.95, 1) / 2
        prof = {}
        for side, step in (("lo", -1), ("hi", +1)):
            x = g.theta[jrole]
            lo, hi = x, x + step * 4.0
            if g.nll_fixed(jrole, hi) + g.ll < crit:
                prof[side] = None
                continue
            for _ in range(20):
                mid = 0.5 * (lo + hi)
                d = g.nll_fixed(jrole, mid) + g.ll
                if d < crit:
                    lo = mid
                else:
                    hi = mid
            prof[side] = round(0.5 * (lo + hi), 4)
        blk["glmm"]["role_profile_ci95"] = [prof["lo"], prof["hi"]]
        sgrid = g.sigma_profile([0.0, 0.05, 0.1, 0.2, 0.3, 0.45, 0.6, 0.8])
        blk["glmm"]["sigma_profile_loglik"] = sgrid
        lr_sig = 2 * (g.ll - sgrid[0.0])
        blk["glmm"]["lrt_sigma_body"] = dict(
            stat=round(float(lr_sig), 4),
            p_boundary=float(0.5 * stats.chi2.sf(max(lr_sig, 0.0), 1)),
            note="50:50 chi2_0/chi2_1 mixture, sigma on the boundary")
        log(f"[{tag}] sigma profile {sgrid}  LRT {lr_sig:.3f}")

        lrts = {}
        for lab, drop in (("role_and_interaction(df2)", ["role", "role_x_renter"]),
                          ("interaction(df1)", ["role_x_renter"])):
            cols = [n for n in names if n not in drop]
            Xr, nmr, _ = design(rows, cols)
            gr = GLMM(rows, Xr).fit()
            stat = 2 * (g.ll - gr.ll)
            lrts[lab] = dict(stat=round(float(stat), 4), df=len(drop),
                             p=float(stats.chi2.sf(stat, len(drop))),
                             loglik_reduced=round(gr.ll, 4))
        # role alone, df 1, against the no-role/no-interaction model
        cols = [n for n in names if n != "role_x_renter"]
        Xa, nma, _ = design(rows, cols)
        ga = GLMM(rows, Xa).fit()
        cols0 = [n for n in cols if n != "role"]
        X0, nm0, _ = design(rows, cols0)
        g0 = GLMM(rows, X0).fit()
        stat = 2 * (ga.ll - g0.ll)
        lrts["role_only(df1, no interaction)"] = dict(
            stat=round(float(stat), 4), df=1, p=float(stats.chi2.sf(stat, 1)),
            loglik_full=round(ga.ll, 4), loglik_reduced=round(g0.ll, 4))
        blk["glmm"]["lrt"] = lrts
        log(f"[{tag}] LRT role df2 {lrts['role_and_interaction(df2)']}")

        bb = BetaBinom(rows, X).fit()
        bse = bb.se()
        phi = float(np.exp(bb.theta[-1]))
        blk["betabinom"] = dict(
            loglik=round(bb.ll, 4), phi=round(phi, 4),
            rho=round(1.0 / (1.0 + phi), 5),
            est={nm: dict(est=float(bb.theta[j]), se=float(bse[j]),
                          lo=float(bb.theta[j] - 1.96 * bse[j]),
                          hi=float(bb.theta[j] + 1.96 * bse[j]))
                 for j, nm in enumerate(names)})
        log(f"[{tag}] beta-binom: role {bb.theta[names.index('role')]:+.3f} "
            f"phi {phi:.2f} rho {1/(1+phi):.4f}")
        res[tag] = blk

    # 4: nested structure
    trows = turn_level(con, staff, cov)
    tfits, tmeta = turn_fits(trows)
    res["turn_level"] = dict(meta=tmeta, fits=tfits)
    log(f"turn-level: {tmeta}")
    for lab, d in tfits.items():
        log(f"  {lab:12s} groups {d['n_groups']:5d} role {d['role']['est']:+.3f} "
            f"se {d['role']['se']:.3f}  renter {d['renter_share']['est']:+.3f} "
            f"se {d['renter_share']['se']:.3f}")

    # 5: LOBO, K-fold, Freedman-Lane -- primary rows
    full, lo = lobo(prim, ["role", "renter_share", "role_x_renter"])
    res["lobo"] = dict(full={k: round(full[k]["est"], 4)
                             for k in ("role", "renter_share", "role_x_renter")},
                       rows=lo)
    with open(OUT / "lobo_primary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(lo[0]))
        w.writeheader(); w.writerows(lo)
    log(f"LOBO done ({len(lo)} bodies)")

    kf, folds = kfold(prim)
    res["kfold"] = dict(folds=folds, results=kf)
    log(f"K-fold: {kf}")

    fl = freedman_lane(prim)
    res["freedman_lane"] = fl
    log(f"Freedman-Lane: {fl}")

    res["runtime_s"] = round(time.time() - t0, 1)
    (OUT / "freq_results.json").write_text(json.dumps(res, indent=2, default=float))

    # flat comparison table
    want = ["role", "renter_share", "role_x_renter", "transit_commute",
            "log_population", "log_income"]
    bay = {}
    for r in csv.DictReader(open(OUT3 / "models" / "model_table.csv")):
        if r["outcome"] == "nimby" and r["spec"] == "R1" and r["paired_only"] == "0":
            key = {"delta_role": "role", "theta_role_x_renter": "role_x_renter"
                   }.get(r["parameter"], r["parameter"])
            bay[key] = (float(r["median"]), float(r["ci95_lo"]), float(r["ci95_hi"]))
    lines = []
    for p in want:
        for tag in ("primary", "secondary"):
            for fit, node in (("GEE exch", "gee_exchangeable"),
                              ("GEE indep", "gee_independence")):
                e = res[tag][node]["est"].get(p)
                if e:
                    lines.append(dict(parameter=p, denominator=tag, fit=fit,
                                      est=round(e["est"], 4), lo=round(e["lo"], 4),
                                      hi=round(e["hi"], 4)))
            for fit, node in (("GLMM", "glmm"), ("beta-binomial", "betabinom")):
                e = res[tag][node]["est"].get(p)
                if e:
                    lines.append(dict(parameter=p, denominator=tag, fit=fit,
                                      est=round(e["est"], 4), lo=round(e["lo"], 4),
                                      hi=round(e["hi"], 4)))
        if p in bay:
            m, l, h = bay[p]
            lines.append(dict(parameter=p, denominator="secondary", fit="Bayesian R1",
                              est=m, lo=l, hi=h))
    with open(OUT / "comparison.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["parameter", "denominator", "fit",
                                           "est", "lo", "hi"])
        w.writeheader(); w.writerows(lines)
    log(f"-> {OUT}  ({res['runtime_s']}s)")


if __name__ == "__main__":
    main()
