#!/usr/bin/env python3
"""
renter_share5_robust.py -- findings 4, 6, 7, 8 and 10 of
plan/REGRESSION-REVIEW-2026-10-19.md, all frequentist, no PyMC.

  LOBO      (finding 6)  leave-one-body-out for every renter_share and
                         transit_commute estimate in runs 2 and 3.
  KFOLD     (finding 7)  grouped 5-fold by body replaces PSIS-LOO; held-out
                         binomial log predictive density per covariate set.
  MULT      (finding 8)  intervals tested vs intervals excluding zero per run,
                         against the count expected under a global null.
  ICC       (finding 4)  episode- and meeting-level ICC of the NIMBY indicator
                         and the implied design effects; turn counts per cluster.
  PERM      (finding 10) Freedman-Lane and whole-covariate-vector body
                         permutations for renter_share and transit_commute.
                         The within-body role swap is NOT re-run (see NOTE).

NOTE on role. plan/RENTER-SHARE-4-SUBSETS.md 4a permutes role by swapping the
two role labels of a body x route cell. That is dropped here as a test of
delta: N_i, y_i and the row-level overdispersion stay attached to the row, so
the swap tests whether the *label* is attached to the larger/more-NIMBY row of
a cell, not delta net of the other terms; and an across-body role permutation
does not exist, because every fitted body has the same role composition
(a dais row and, where it clears the floor, a public row). delta's evidence is
therefore the model plus the by-body cluster-robust SE, and nothing else.

Writes output/renter_share_5/robust/. Read-only on the DB and on runs 2-4,
no network, no spend, nothing committed.

    nohup .venv/bin/python analysis/renter_share5_robust.py \
        > output/renter_share_5/robust.log 2>&1 &
"""
import collections
import csv
import json
import pathlib
import re
import sqlite3

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT2 = ROOT / "output" / "renter_share_2"
OUT3 = ROOT / "output" / "renter_share_3"
OUT = ROOT / "output" / "renter_share_5" / "robust"
DB = "file:" + str(pathlib.Path.home() / ".local/share/ytpt/state.sqlite") + "?mode=ro"
CLUSTERS = (ROOT / "output" / "stance_s1" / "nimby_view" /
            "speaker_clusters_roleaudit.csv")
ROUTE_REF = "classifier"
NPERM = 5000
NFOLD = 5
SEED = 20261019
KEPT8 = ["renter_share", "log_income", "log_density", "built_2010_plus",
         "nhwhite_share", "transit_commute", "vacancy_rate", "log_population"]


# ------------------------------------------------------------------ IRLS core
def irls(X, y, N, maxit=300, tol=1e-11):
    X = np.asarray(X, float)
    y, N = np.asarray(y, float), np.asarray(N, float)
    k = X.shape[1]
    beta, ok = np.zeros(k), False
    for _ in range(maxit):
        eta = np.clip(X @ beta, -30, 30)
        p = 1 / (1 + np.exp(-eta))
        W = np.maximum(N * p * (1 - p), 1e-9)
        z = eta + (y - N * p) / W
        A = (X.T * W) @ X
        new = np.linalg.pinv(A) @ ((X.T * W) @ z)
        if not np.all(np.isfinite(new)):
            break
        if np.max(np.abs(new - beta)) < tol:
            beta, ok = new, True
            break
        beta = new
    return beta, ok


def irls_se(X, y, N, groups):
    """beta plus HC3 and by-body cluster-robust SEs (run 3 4's routine)."""
    X = np.asarray(X, float)
    y, N = np.asarray(y, float), np.asarray(N, float)
    n, k = X.shape
    beta, ok = irls(X, y, N)
    eta = np.clip(X @ beta, -30, 30)
    p = 1 / (1 + np.exp(-eta))
    W = np.maximum(N * p * (1 - p), 1e-9)
    Ai = np.linalg.pinv((X.T * W) @ X)
    Xw = X * np.sqrt(W)[:, None]
    h = np.einsum("ij,jk,ik->i", Xw, np.linalg.pinv(Xw.T @ Xw), Xw)
    s = y - N * p
    V3 = Ai @ ((X.T * (s ** 2 / np.maximum((1 - h) ** 2, 1e-12))) @ X) @ Ai
    gs = sorted(set(groups))
    meat = np.zeros((k, k))
    g = np.asarray(groups)
    for q in gs:
        m = g == q
        sg = X[m].T @ s[m]
        meat += np.outer(sg, sg)
    G = len(gs)
    Vc = (G / max(G - 1, 1)) * ((n - 1) / max(n - k, 1)) * (Ai @ meat @ Ai)
    return (beta, np.sqrt(np.maximum(np.diag(V3), 0)),
            np.sqrt(np.maximum(np.diag(Vc), 0)), ok, G)


def build_X(rows, cols, role=False, ixr=False, ixy=False, routes=None,
            mu=None, sd=None):
    """standardised covariates + optional role terms + route dummies + 1."""
    names, blocks = [], []
    if cols:
        Xr = np.column_stack([[r[c] for r in rows] for c in cols]).astype(float)
        if mu is None:
            mu, sd = Xr.mean(0), Xr.std(0, ddof=0)
            sd = np.where(sd > 0, sd, 1.0)
        Z = (Xr - mu) / sd
        blocks.append(Z)
        names += list(cols)
    else:
        Z, mu, sd = None, None, None
    rc = np.array([r.get("role_code", 0) for r in rows], float)
    if role:
        blocks.append(rc[:, None])
        names.append("role")
    if ixr:
        blocks.append((rc * Z[:, cols.index("renter_share")])[:, None])
        names.append("role_x_renter")
    if ixy:
        blocks.append((rc * Z[:, cols.index("log_income")])[:, None])
        names.append("role_x_income")
    if routes is None:
        routes = sorted({r["route"] for r in rows},
                        key=lambda s: (s != ROUTE_REF, s))
    for i, rt in enumerate(routes):
        if i == 0:
            continue
        blocks.append(np.array([1.0 if r["route"] == rt else 0.0
                                for r in rows])[:, None])
        names.append(f"route[{rt}]")
    blocks.append(np.ones((len(rows), 1)))
    names.append("intercept")
    return np.column_stack(blocks), names, routes, mu, sd


# ----------------------------------------------------------------- the data
def load_run2():
    rows = []
    for r in csv.DictReader(open(OUT2 / "outcomes_clusters.csv")):
        d = dict(route=r["route"], body=r["body"])
        for c in KEPT8:
            d[c] = float(r[c])
        for c in ("pub_n", "pub_nimby", "pub_yimby", "all_n", "all_nimby",
                  "all_yimby", "dais_n", "dais_nimby", "dais_yimby",
                  "pub_neutral"):
            d[c] = int(r[c])
        rows.append(d)
    return rows


def load_run3():
    rows = []
    for r in csv.DictReader(open(OUT3 / "outcomes_role_stacked.csv")):
        d = dict(route=r["route"], body=r["body"], role=r["role"],
                 role_code=int(r["role_code"]), paired=int(r["paired_cell"]))
        for c in KEPT8:
            d[c] = float(r[c])
        for c in ("n", "nimby", "yimby", "neutral", "episodes"):
            d[c] = int(r[c])
        rows.append(d)
    return rows


R2_OUTCOMES = [("public NIMBY", "pub_nimby", "pub_n"),
               ("public YIMBY", "pub_yimby", "pub_n"),
               ("all-roles NIMBY", "all_nimby", "all_n"),
               ("all-roles YIMBY", "all_yimby", "all_n")]
R2_SPECS = [("(a) base", ["renter_share", "log_income"]),
            ("(b) base+density", ["renter_share", "log_income", "log_density"]),
            ("(c) all eight (unpenalised)", KEPT8)]


# ------------------------------------------------------------ finding 6: LOBO
def lobo():
    """Leave-one-body-out for renter_share and transit_commute."""
    out = []
    r2 = load_run2()
    for oname, ycol, ncol in R2_OUTCOMES:
        rows = [r for r in r2 if r[ncol] >= 5]
        for sname, cols in R2_SPECS:
            X, names, routes, mu, sd = build_X(rows, cols)
            y = [r[ycol] for r in rows]
            N = [r[ncol] for r in rows]
            b0, _ = irls(X, y, N)
            for term in ("renter_share", "transit_commute"):
                if term not in names:
                    continue
                j = names.index(term)
                drops = []
                for body in sorted({r["body"] for r in rows}):
                    sub = [r for r in rows if r["body"] != body]
                    if len({r["body"] for r in sub}) < 3:
                        continue
                    Xs, ns, _, _, _ = build_X(sub, cols)
                    bs, ok = irls(Xs, [r[ycol] for r in sub],
                                  [r[ncol] for r in sub])
                    if ok and term in ns:
                        drops.append((body, float(bs[ns.index(term)])))
                vals = np.array([v for _, v in drops])
                lo = min(drops, key=lambda t: t[1])
                hi = max(drops, key=lambda t: t[1])
                far = max(drops, key=lambda t: abs(t[1] - b0[j]))
                out.append(dict(
                    run="run2", outcome=oname, spec=sname, term=term,
                    n_rows=len(rows), beta_full=float(b0[j]),
                    lobo_min_body=lo[0], lobo_min=lo[1],
                    lobo_max_body=hi[0], lobo_max=hi[1],
                    lobo_range=float(vals.max() - vals.min()),
                    worst_body=far[0], worst_beta=far[1],
                    worst_pct_move=(float(abs(far[1] - b0[j])
                                          / max(abs(b0[j]), 1e-9)) * 100),
                    sign_flips=int(((vals * b0[j]) < 0).sum())))
    # run 3 stacked role design
    r3 = load_run3()
    for ycol, oname in (("nimby", "NIMBY"), ("yimby", "YIMBY")):
        for sname, ixy in (("R1", False), ("R2", True)):
            X, names, routes, mu, sd = build_X(r3, KEPT8, role=True, ixr=True,
                                               ixy=ixy)
            y = [r[ycol] for r in r3]
            N = [r["n"] for r in r3]
            b0, _ = irls(X, y, N)
            jr, jx = names.index("renter_share"), names.index("role_x_renter")
            jt = names.index("transit_commute")
            targets = {"renter_share (dais slope)": lambda b: b[jr],
                       "renter_share (public slope)": lambda b: b[jr] + b[jx],
                       "transit_commute": lambda b: b[jt],
                       "role (delta)": lambda b: b[names.index("role")]}
            full = {k: float(f(b0)) for k, f in targets.items()}
            drops = {k: [] for k in targets}
            for body in sorted({r["body"] for r in r3}):
                sub = [r for r in r3 if r["body"] != body]
                Xs, ns, _, _, _ = build_X(sub, KEPT8, role=True, ixr=True,
                                          ixy=ixy)
                bs, ok = irls(Xs, [r[ycol] for r in sub],
                              [r["n"] for r in sub])
                if not ok or ns != names:
                    continue
                for k, f in targets.items():
                    drops[k].append((body, float(f(bs))))
            for k in targets:
                d = drops[k]
                vals = np.array([v for _, v in d])
                lo = min(d, key=lambda t: t[1])
                hi = max(d, key=lambda t: t[1])
                far = max(d, key=lambda t: abs(t[1] - full[k]))
                out.append(dict(
                    run="run3", outcome=oname, spec=sname, term=k,
                    n_rows=len(r3), beta_full=full[k],
                    lobo_min_body=lo[0], lobo_min=lo[1],
                    lobo_max_body=hi[0], lobo_max=hi[1],
                    lobo_range=float(vals.max() - vals.min()),
                    worst_body=far[0], worst_beta=far[1],
                    worst_pct_move=(float(abs(far[1] - full[k])
                                          / max(abs(full[k]), 1e-9)) * 100),
                    sign_flips=int(((vals * full[k]) < 0).sum())))
    return out


# ------------------------------------------------- finding 7: grouped K-fold
KFOLD_SETS = [("route only", []),
              ("renter+income", ["renter_share", "log_income"]),
              ("+density", ["renter_share", "log_income", "log_density"]),
              ("all eight", KEPT8)]


def binom_lpd(y, N, p):
    from math import lgamma, log
    p = float(min(max(p, 1e-12), 1 - 1e-12))
    return (lgamma(N + 1) - lgamma(y + 1) - lgamma(N - y + 1)
            + y * log(p) + (N - y) * log(1 - p))


def kfold():
    rng = np.random.default_rng(SEED)
    out = []
    r2 = load_run2()
    for oname, ycol, ncol in R2_OUTCOMES:
        rows = [r for r in r2 if r[ncol] >= 5]
        bodies = sorted({r["body"] for r in rows})
        order = list(rng.permutation(len(bodies)))
        fold_of = {bodies[order[i]]: i % NFOLD for i in range(len(bodies))}
        allroutes = sorted({r["route"] for r in rows},
                           key=lambda s: (s != ROUTE_REF, s))
        for sname, cols in KFOLD_SETS:
            elpd, pw, nfail = 0.0, [], 0
            for f in range(NFOLD):
                tr = [r for r in rows if fold_of[r["body"]] != f]
                te = [r for r in rows if fold_of[r["body"]] == f]
                Xtr, names, _, mu, sd = build_X(tr, cols, routes=allroutes)
                b, ok = irls(Xtr, [r[ycol] for r in tr],
                             [r[ncol] for r in tr])
                if not ok:
                    nfail += 1
                Xte, _, _, _, _ = build_X(te, cols, routes=allroutes,
                                          mu=mu, sd=sd)
                eta = np.clip(Xte @ b, -30, 30)
                p = 1 / (1 + np.exp(-eta))
                for i, r in enumerate(te):
                    v = binom_lpd(r[ycol], r[ncol], p[i])
                    pw.append(v)
                    elpd += v
            pw = np.array(pw)
            out.append(dict(run="run2", outcome=oname, cov_set=sname,
                            n_params=len(cols) + len(allroutes),
                            n_cells=len(rows), folds=NFOLD,
                            elpd=float(elpd),
                            se=float(np.sqrt(len(pw)) * pw.std(ddof=1)),
                            worst_cell_lpd=float(pw.min()),
                            nonconverged_folds=nfail))
    # same on run 3's stacked rows, role always in
    r3 = load_run3()
    bodies = sorted({r["body"] for r in r3})
    order = list(rng.permutation(len(bodies)))
    fold_of = {bodies[order[i]]: i % NFOLD for i in range(len(bodies))}
    allroutes = sorted({r["route"] for r in r3},
                       key=lambda s: (s != ROUTE_REF, s))
    for ycol, oname in (("nimby", "NIMBY"), ("yimby", "YIMBY")):
        for sname, cols in KFOLD_SETS:
            for ixr in (False, True):
                if ixr and "renter_share" not in cols:
                    continue
                elpd, pw = 0.0, []
                for f in range(NFOLD):
                    tr = [r for r in r3 if fold_of[r["body"]] != f]
                    te = [r for r in r3 if fold_of[r["body"]] == f]
                    Xtr, names, _, mu, sd = build_X(tr, cols, role=True,
                                                    ixr=ixr, routes=allroutes)
                    b, ok = irls(Xtr, [r[ycol] for r in tr],
                                 [r["n"] for r in tr])
                    Xte, _, _, _, _ = build_X(te, cols, role=True, ixr=ixr,
                                              routes=allroutes, mu=mu, sd=sd)
                    p = 1 / (1 + np.exp(-np.clip(Xte @ b, -30, 30)))
                    for i, r in enumerate(te):
                        v = binom_lpd(r[ycol], r["n"], p[i])
                        pw.append(v)
                        elpd += v
                pw = np.array(pw)
                out.append(dict(
                    run="run3", outcome=oname,
                    cov_set=sname + (" + role x renter" if ixr else "")
                    + " + role",
                    n_params=len(cols) + len(allroutes) + 1 + int(ixr),
                    n_cells=len(r3), folds=NFOLD, elpd=float(elpd),
                    se=float(np.sqrt(len(pw)) * pw.std(ddof=1)),
                    worst_cell_lpd=float(pw.min()), nonconverged_folds=0))
    return out


# ------------------------------------------------- finding 8: multiplicity
SUBST_SKIP = re.compile(r"^(alpha|intercept|sigma_|gamma\[|slope_renter_|"
                        r"odds_ratio_)")


def multiplicity():
    out = []
    for run, path in (("run2", OUT2 / "models" / "model_table.csv"),
                      ("run3", OUT3 / "models" / "model_table.csv")):
        rows = list(csv.DictReader(open(path)))
        pkey = "parameter" if "parameter" in rows[0] else "covariate"
        skey = "spec" if "spec" in rows[0] else "model"
        sub = []
        for r in rows:
            if SUBST_SKIP.match(r[pkey]):
                continue
            q = dict(parameter=r[pkey], spec=r[skey], outcome=r["outcome"],
                     paired_only=r.get("paired_only", "0"))
            q["excl_zero_90"] = int(r["excl_zero_90"])
            q["excl_zero_95"] = int(r["excl_zero_95"]) if "excl_zero_95" in r \
                else int(float(r["ci95_lo"]) > 0 or float(r["ci95_hi"]) < 0)
            sub.append(q)
        n90 = sum(r["excl_zero_90"] for r in sub)
        n95 = sum(r["excl_zero_95"] for r in sub)
        byfit = collections.Counter((r["outcome"], r["spec"],
                                     r["paired_only"]) for r in sub)
        out.append(dict(run=run, intervals=len(sub), fits=len(byfit),
                        excl_90=n90, excl_95=n95,
                        expected_90=round(0.10 * len(sub), 1),
                        expected_95=round(0.05 * len(sub), 1),
                        distinct_params=len({r["parameter"] for r in sub}),
                        params_excl_90=sorted({r["parameter"] for r in sub
                                               if r["excl_zero_90"]}),
                        params_excl_95=sorted({r["parameter"] for r in sub
                                               if r["excl_zero_95"]}),
                        rows_excl_90=sorted(f"{r['outcome']}/{r['spec']}/"
                                            f"{r['parameter']}" for r in sub
                                            if r["excl_zero_90"]),
                        rows_excl_95=sorted(f"{r['outcome']}/{r['spec']}/"
                                            f"{r['parameter']}" for r in sub
                                            if r["excl_zero_95"])))
    return out


# ------------------------------------------------------- finding 4: ICC
def icc_block(groups, vals):
    """one-way random-effects ICC (ANOVA) + mean group size + design effect."""
    g = collections.defaultdict(list)
    for k, v in zip(groups, vals):
        g[k].append(v)
    k = len(g)
    n = sum(len(v) for v in g.values())
    if k < 2 or n <= k:
        return None
    gm = np.mean(vals)
    ssb = sum(len(v) * (np.mean(v) - gm) ** 2 for v in g.values())
    ssw = sum(sum((x - np.mean(v)) ** 2 for x in v) for v in g.values())
    msb, msw = ssb / (k - 1), ssw / (n - k)
    sizes = np.array([len(v) for v in g.values()], float)
    n0 = (n - (sizes ** 2).sum() / n) / (k - 1)
    var_b = max((msb - msw) / max(n0, 1e-9), 0.0)
    icc = var_b / (var_b + msw) if (var_b + msw) > 0 else 0.0
    mbar = n / k
    return dict(groups=k, n=n, mean_size=round(mbar, 2), icc=round(icc, 4),
                design_effect=round(1 + (mbar - 1) * icc, 3),
                se_inflation=round(np.sqrt(1 + (mbar - 1) * icc), 3))


def meeting_of(ep):
    return re.sub(r"_e\d+$", "", ep)


def icc():
    rows = [r for r in csv.DictReader(open(CLUSTERS))
            if r["label"] != "no stance" and r["role"] in ("dais", "public")]
    out = []
    for name, sel in (("all stance-bearing", rows),
                      ("dais", [r for r in rows if r["role"] == "dais"]),
                      ("public", [r for r in rows if r["role"] == "public"])):
        nim = [1.0 if r["label"] == "NIMBY" else 0.0 for r in sel]
        dirsel = [r for r in sel if r["label"] in ("NIMBY", "YIMBY")]
        nimd = [1.0 if r["label"] == "NIMBY" else 0.0 for r in dirsel]
        for lvl, key, s, v in (
                ("episode", "episode_id", sel, nim),
                ("meeting", "meeting", sel, nim),
                ("episode (directional)", "episode_id", dirsel, nimd),
                ("meeting (directional)", "meeting", dirsel, nimd)):
            gk = [(r["episode_id"] if "episode" in lvl else
                   meeting_of(r["episode_id"])) for r in s]
            b = icc_block(gk, v)
            if b:
                out.append(dict(subset=name, level=lvl, outcome="NIMBY", **b))
    turns = collections.Counter()
    for name, sel in (("all stance-bearing", rows),
                      ("dais", [r for r in rows if r["role"] == "dais"]),
                      ("public", [r for r in rows if r["role"] == "public"])):
        nt = np.array([int(r["n_turns"]) for r in sel], float)
        wd = np.array([int(r["words"]) for r in sel], float)
        turns[name] = dict(
            clusters=len(sel), one_turn_pct=round(100 * (nt == 1).mean(), 1),
            median_turns=float(np.median(nt)),
            words_p10=float(np.percentile(wd, 10)),
            words_median=float(np.median(wd)),
            words_p90=float(np.percentile(wd, 90)))
    return out, dict(turns)


# ---------------------------------------------- finding 10: permutation nulls
def perm_body(rows, cols, ixr, ycol, Ncol, term, mode, nperm=NPERM):
    """Freedman-Lane and whole-vector body permutations of a partial slope.

    mode='whole_vector': permute the entire body-level covariate vector across
      bodies (every row of a body moves together), so the collinearity
      structure among the eight covariates is preserved exactly and only the
      covariate-to-outcome link is broken.
    mode='freedman_lane': IRLS working response z and weights W from the FULL
      fit; regress z on X without `term` (weighted); permute the weighted
      residuals in body blocks; rebuild z* = fitted_reduced + permuted resid;
      refit the full weighted least squares and read the coefficient on `term`.
    """
    rng = np.random.default_rng(SEED)
    X, names, routes, mu, sd = build_X(rows, cols, role=True, ixr=ixr)
    y = np.array([r[ycol] for r in rows], float)
    N = np.array([r[Ncol] for r in rows], float)
    beta, se3, sec, ok, G = irls_se(X, y, N, [r["body"] for r in rows])
    j = names.index(term)
    obs = float(beta[j])
    bodies = sorted({r["body"] for r in rows})
    bidx = np.array([bodies.index(r["body"]) for r in rows])

    draws = []
    if mode == "whole_vector":
        bodyvec = {b: {c: next(r[c] for r in rows if r["body"] == b)
                       for c in cols} for b in bodies}
        for _ in range(nperm):
            pm_ = rng.permutation(len(bodies))
            mapped = {bodies[i]: bodies[pm_[i]] for i in range(len(bodies))}
            prows = []
            for r in rows:
                q = dict(r)
                q.update(bodyvec[mapped[r["body"]]])
                prows.append(q)
            Xp, npn, _, _, _ = build_X(prows, cols, role=True, ixr=ixr)
            bp, okp = irls(Xp, y, N)
            if okp:
                draws.append(float(bp[npn.index(term)]))
    elif mode == "freedman_lane":
        eta = np.clip(X @ beta, -30, 30)
        p = 1 / (1 + np.exp(-eta))
        W = np.maximum(N * p * (1 - p), 1e-9)
        z = eta + (y - N * p) / W
        keep = [i for i in range(len(names)) if i != j]
        Xr = X[:, keep]
        sw = np.sqrt(W)
        br = np.linalg.pinv((Xr.T * W) @ Xr) @ ((Xr.T * W) @ z)
        fitr = Xr @ br
        res = (z - fitr) * sw            # weighted residuals, exchangeable
        A = np.linalg.pinv((X.T * W) @ X)
        for _ in range(nperm):
            pb = rng.permutation(len(bodies))
            newres = np.empty_like(res)
            for i in range(len(rows)):
                src = pb[bidx[i]]
                pool = np.where(bidx == src)[0]
                newres[i] = res[pool[rng.integers(len(pool))]]
            zs = fitr + newres / sw
            bs = A @ ((X.T * W) @ zs)
            draws.append(float(bs[j]))
    draws = np.array(draws)
    ge = int((np.abs(draws) >= abs(obs)).sum())
    return dict(term=term, mode=mode, beta_obs=obs,
                se_model=float(se3[j]), se_cluster=float(sec[j]),
                n_perm=len(draws), n_ge=ge,
                p_two_sided=round((1 + ge) / (1 + len(draws)), 4),
                null_sd=float(draws.std(ddof=1)),
                null_q025=float(np.percentile(draws, 2.5)),
                null_median=float(np.median(draws)),
                null_q975=float(np.percentile(draws, 97.5)))


def permutations():
    r3 = load_run3()
    out = []
    for ycol, oname in (("nimby", "NIMBY"), ("yimby", "YIMBY")):
        for dname, cols, ixr in (("FULL (8 cov + role x renter)", KEPT8, True),
                                 ("MIN (renter + role x renter)",
                                  ["renter_share", "log_income"], True)):
            for term in ("renter_share", "transit_commute"):
                if term not in cols:
                    continue
                for mode in ("whole_vector", "freedman_lane"):
                    r = perm_body(r3, cols, ixr, ycol, "n", term, mode)
                    r.update(outcome=oname, design=dname, run="run3")
                    out.append(r)
                    print(f"  {oname:6s} {dname:28s} {term:16s} {mode:14s} "
                          f"beta {r['beta_obs']:+.4f} p {r['p_two_sided']:.4f}")
    # the same two nulls on run 2's public cells, base and all-eight designs
    r2 = load_run2()
    for oname, ycol, ncol in R2_OUTCOMES:
        rows = [r for r in r2 if r[ncol] >= 5]
        for dname, cols in (("(a) base", ["renter_share", "log_income"]),
                            ("(c) all eight", KEPT8)):
            for term in ("renter_share", "transit_commute"):
                if term not in cols:
                    continue
                for mode in ("whole_vector", "freedman_lane"):
                    r = perm_body(rows, cols, False, ycol, ncol, term, mode)
                    r.update(outcome=oname, design=dname, run="run2")
                    out.append(r)
                    print(f"  {oname:16s} {dname:14s} {term:16s} {mode:14s} "
                          f"beta {r['beta_obs']:+.4f} p {r['p_two_sided']:.4f}")
    return out


# ----------------------------------- run 3 cluster-robust SEs, as primary
def run3_cluster_primary():
    r3 = load_run3()
    out = []
    for ycol, oname in (("nimby", "NIMBY"), ("yimby", "YIMBY")):
        for sname, ixy in (("R1", False), ("R2", True)):
            X, names, _, _, sd = build_X(r3, KEPT8, role=True, ixr=True,
                                         ixy=ixy)
            b, se3, sec, ok, G = irls_se(X, [r[ycol] for r in r3],
                                         [r["n"] for r in r3],
                                         [r["body"] for r in r3])
            for j, nm in enumerate(names):
                if nm == "intercept":
                    continue
                out.append(dict(
                    outcome=oname, spec=sname, term=nm, beta=float(b[j]),
                    se_cluster=float(sec[j]),
                    cl_lo90=float(b[j] - 1.645 * sec[j]),
                    cl_hi90=float(b[j] + 1.645 * sec[j]),
                    cl_lo95=float(b[j] - 1.96 * sec[j]),
                    cl_hi95=float(b[j] + 1.96 * sec[j]),
                    se_hc3=float(se3[j]),
                    ratio_cluster_hc3=float(sec[j] / max(se3[j], 1e-9)),
                    n_body_clusters=G))
            jr, jx = names.index("renter_share"), names.index("role_x_renter")
            out.append(dict(
                outcome=oname, spec=sname, term="renter_share (public slope)",
                beta=float(b[jr] + b[jx]), se_cluster=float("nan"),
                cl_lo90=float("nan"), cl_hi90=float("nan"),
                cl_lo95=float("nan"), cl_hi95=float("nan"),
                se_hc3=float("nan"), ratio_cluster_hc3=float("nan"),
                n_body_clusters=G))
    return out


def wcsv(path, recs):
    if not recs:
        return
    keys = list(recs[0])
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in recs:
            w.writerow({k: (json.dumps(v) if isinstance(v, (list, dict))
                            else v) for k, v in r.items()})


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print("== finding 6: leave-one-body-out")
    L = lobo()
    wcsv(OUT / "lobo.csv", L)
    for r in L:
        print(f"  {r['run']} {r['outcome']:16s} {r['spec']:28s} {r['term']:28s}"
              f" full {r['beta_full']:+.4f} -> worst {r['worst_beta']:+.4f} "
              f"(drop {r['worst_body']}, {r['worst_pct_move']:.0f} %) "
              f"flips {r['sign_flips']}")

    print("\n== finding 7: grouped 5-fold by body")
    K = kfold()
    wcsv(OUT / "kfold.csv", K)
    for r in K:
        print(f"  {r['run']} {r['outcome']:16s} {r['cov_set']:34s} "
              f"ELPD {r['elpd']:+9.2f} (se {r['se']:6.2f}) p {r['n_params']}")

    print("\n== finding 8: multiplicity")
    M = multiplicity()
    wcsv(OUT / "multiplicity.csv", M)
    for r in M:
        print(f"  {r['run']}: {r['intervals']} intervals over {r['fits']} "
              f"fits; 90 % exclusions {r['excl_90']} (expected "
              f"{r['expected_90']}), 95 % {r['excl_95']} (expected "
              f"{r['expected_95']})")
        print(f"     90 %: {r['params_excl_90']}")
        print(f"     95 %: {r['params_excl_95']}")

    print("\n== finding 4: ICC and design effects")
    I, T = icc()
    wcsv(OUT / "icc.csv", I)
    (OUT / "cluster_size.json").write_text(json.dumps(T, indent=2))
    for r in I:
        print(f"  {r['subset']:20s} {r['level']:24s} ICC {r['icc']:.4f} "
              f"m-bar {r['mean_size']:5.2f} deff {r['design_effect']:.3f} "
              f"SE x{r['se_inflation']:.3f} ({r['groups']} groups, n {r['n']})")
    for k, v in T.items():
        print(f"  {k:20s} clusters {v['clusters']:5d} one-turn "
              f"{v['one_turn_pct']:.1f} % words p10/median/p90 "
              f"{v['words_p10']:.0f}/{v['words_median']:.0f}/{v['words_p90']:.0f}")

    print("\n== run 3 by-body cluster-robust SEs, as primary")
    C = run3_cluster_primary()
    wcsv(OUT / "run3_cluster_primary.csv", C)
    for r in C:
        if r["term"] in ("role", "renter_share", "transit_commute",
                         "role_x_renter", "renter_share (public slope)"):
            print(f"  {r['outcome']:6s} {r['spec']} {r['term']:28s} "
                  f"{r['beta']:+.4f} cluster SE {r['se_cluster']:.4f} "
                  f"95 % [{r['cl_lo95']:+.4f}, {r['cl_hi95']:+.4f}]")

    print(f"\n== finding 10: permutation nulls ({NPERM} draws each)")
    P = permutations()
    wcsv(OUT / "permutation_fl.csv", P)

    (OUT / "results.json").write_text(json.dumps(dict(
        seed=SEED, n_perm=NPERM, n_fold=NFOLD, kept=KEPT8,
        role_permutation_note=(
            "The within-body role swap of plan/RENTER-SHARE-4-SUBSETS.md 4a is "
            "NOT re-run as a test of delta: N_i, y_i and the row-level "
            "overdispersion stay attached to the row, so the swap tests "
            "whether the label attaches to the larger / more-NIMBY row of a "
            "cell rather than delta net of the other terms. An across-body "
            "role permutation does not exist here: every fitted body has the "
            "same role composition, so permuting role across bodies is the "
            "identity. delta's evidence is the model plus the by-body "
            "cluster-robust SE."),
        lobo=L, kfold=K, multiplicity=M, icc=I, cluster_size=T,
        run3_cluster_primary=C, permutation=P), indent=2, default=str))
    print(f"\n-> {OUT}")


if __name__ == "__main__":
    main()
