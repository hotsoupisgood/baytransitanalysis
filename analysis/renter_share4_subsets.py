#!/usr/bin/env python3
"""RENTER-SHARE-4 — all-subsets model comparison + permutation tests.

Reads output/renter_share_3/outcomes_role_stacked.csv (64 stacked
body x route x role rows, floor 5 clusters with a stance per row) and
output/renter_share_2/screen.json (run 2's kept covariate set, re-used
unchanged). Writes output/renter_share_4/.

EXPLORATORY. No YouTube request, no GPU, no API call, no spend, no DB write.
.env, plan/frame.csv, plan/DECISIONS.md untouched. Nothing committed.

Part 1 - all subsets
  Always in: intercept, role (dais 0 / public 1), route dummies (ref
  classifier; route is a stratum in every model, never pooled).
  Enumerated: every subset of run 2's 8 kept covariates, x optionally the
  role x renter_share interaction when renter_share is in the subset
  => 2^8 + 2^7 = 384 models per outcome.
  Each fit: binomial logistic IRLS. Overdispersion is handled as
  quasi-binomial: c_hat = Pearson chi2 / df from the GLOBAL model (all 8
  covariates + interaction), applied to every model, per Burnham & Anderson.
  QAICc = -2logL/c_hat + 2K + 2K(K+1)/(n-K-1), QBIC = -2logL/c_hat + K ln n,
  K = n_params + 1 (the dispersion). Plain AICc/BIC (c_hat = 1) also kept.
  Akaike weights from QAICc; BMA weights from QBIC. Model-averaged
  coefficients use zero-substitution (full-model averaging) with Buckland
  unconditional SEs.

Part 2 - permutation tests
  (a) renter share and role x renter: permute renter_share ACROSS BODIES,
      every row of a body moving together, in two designs -
      FULL (role + 8 covariates + interaction + routes) and
      MIN  (role + renter_share + interaction + routes).
  (b) role main effect: swap the role labels of the two rows of a
      body x route cell (29 cells carry both roles), which is the within-body
      randomisation the design admits; shows role is not a labelling artefact.
  Two-sided p = (1 + #{|b_perm| >= |b_obs|}) / (1 + n_perm).
"""
import csv
import json
import itertools
import math
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
IN3 = ROOT / "output" / "renter_share_3"
IN2 = ROOT / "output" / "renter_share_2"
OUT = ROOT / "output" / "renter_share_4"
OUT.mkdir(parents=True, exist_ok=True)

COVS = ["renter_share", "log_income", "log_density", "built_2010_plus",
        "nhwhite_share", "transit_commute", "vacancy_rate", "log_population"]
ROUTE_REF = "classifier"
OUTCOMES = ["nimby", "yimby"]
NPERM = 5000
SEED = 20261019

SHORT = {"renter_share": "renter", "log_income": "income",
         "log_density": "density", "built_2010_plus": "built10",
         "nhwhite_share": "nhwhite", "transit_commute": "transit",
         "vacancy_rate": "vacancy", "log_population": "logpop"}


# --------------------------------------------------------------- data
def load():
    rows = []
    for r in csv.DictReader(open(IN3 / "outcomes_role_stacked.csv")):
        d = dict(route=r["route"], body=r["body"], role=r["role"],
                 role_code=int(r["role_code"]), paired=int(r["paired_cell"]),
                 n=int(r["n"]), nimby=int(r["nimby"]), yimby=int(r["yimby"]))
        for c in COVS:
            d[c] = float(r[c])
        rows.append(d)
    return rows


def irls(X, y, N, maxit=200, tol=1e-11):
    """Binomial logistic IRLS. Returns beta, cov (model-based), p, ok."""
    X = np.asarray(X, float)
    n, k = X.shape
    beta, ok = np.zeros(k), False
    for _ in range(maxit):
        eta = np.clip(X @ beta, -30, 30)
        p = 1.0 / (1.0 + np.exp(-eta))
        W = np.maximum(N * p * (1 - p), 1e-9)
        z = eta + (y - N * p) / W
        A = (X.T * W) @ X
        new = np.linalg.pinv(A) @ ((X.T * W) @ z)
        if np.max(np.abs(new - beta)) < tol:
            beta, ok = new, True
            break
        beta = new
    eta = np.clip(X @ beta, -30, 30)
    p = 1.0 / (1.0 + np.exp(-eta))
    W = np.maximum(N * p * (1 - p), 1e-9)
    cov = np.linalg.pinv((X.T * W) @ X)
    return beta, cov, p, bool(ok)


def beta_only(X, y, N, maxit=120, tol=1e-9):
    X = np.asarray(X, float)
    beta = np.zeros(X.shape[1])
    for _ in range(maxit):
        eta = np.clip(X @ beta, -30, 30)
        p = 1.0 / (1.0 + np.exp(-eta))
        W = np.maximum(N * p * (1 - p), 1e-9)
        z = eta + (y - N * p) / W
        new = np.linalg.pinv((X.T * W) @ X) @ ((X.T * W) @ z)
        if np.max(np.abs(new - beta)) < tol:
            return new
        beta = new
    return beta


def binom_ll(y, N, p):
    """Binomial log-likelihood including the constant."""
    p = np.clip(p, 1e-12, 1 - 1e-12)
    out = 0.0
    for yi, Ni, pi in zip(y, N, p):
        out += (math.lgamma(Ni + 1) - math.lgamma(yi + 1)
                - math.lgamma(Ni - yi + 1)
                + yi * math.log(pi) + (Ni - yi) * math.log1p(-pi))
    return out


def pearson_chi2(y, N, p):
    p = np.clip(p, 1e-10, 1 - 1e-10)
    return float(np.sum((y - N * p) ** 2 / (N * p * (1 - p))))


def cluster_se(X, y, N, p, groups):
    """By-body cluster-robust sandwich with the usual finite-sample factor."""
    n, k = X.shape
    W = np.maximum(N * p * (1 - p), 1e-9)
    Ai = np.linalg.pinv((X.T * W) @ X)
    s = y - N * p
    gs = sorted(set(groups))
    meat = np.zeros((k, k))
    ga = np.asarray(groups)
    for g in gs:
        m = ga == g
        sg = X[m].T @ s[m]
        meat += np.outer(sg, sg)
    G = len(gs)
    corr = (G / max(G - 1, 1)) * ((n - 1) / max(n - k, 1))
    V = corr * (Ai @ meat @ Ai)
    return np.sqrt(np.maximum(np.diag(V), 0)), G


# --------------------------------------------------------- design blocks
def build(rows):
    routes = sorted({r["route"] for r in rows})
    dummies = [q for q in routes if q != ROUTE_REF]
    Zraw = np.column_stack([[r[c] for r in rows] for c in COVS]).astype(float)
    mu, sd = Zraw.mean(0), Zraw.std(0, ddof=0)
    sd = np.where(sd > 0, sd, 1.0)
    Z = (Zraw - mu) / sd
    role = np.array([r["role_code"] for r in rows], float)
    base = [np.ones(len(rows)), role]
    base_names = ["intercept", "role"]
    for q in dummies:
        base.append(np.array([1.0 if r["route"] == q else 0.0 for r in rows]))
        base_names.append(f"route[{q}]")
    base = np.column_stack(base)
    N = np.array([r["n"] for r in rows], float)
    ys = {o: np.array([r[o] for r in rows], float) for o in OUTCOMES}
    bodies = [r["body"] for r in rows]
    return dict(base=base, base_names=base_names, Z=Z, role=role, N=N, ys=ys,
                bodies=bodies, mu=mu, sd=sd, routes=routes, dummies=dummies)


def assemble(D, subset, ix, Zcol=None):
    """Design for covariate subset `subset` (+ role x renter if ix)."""
    Z = D["Z"] if Zcol is None else Zcol
    cols = [D["base"]]
    names = list(D["base_names"])
    for c in subset:
        cols.append(Z[:, COVS.index(c)][:, None])
        names.append(c)
    if ix:
        cols.append((D["role"] * Z[:, COVS.index("renter_share")])[:, None])
        names.append("role:renter_share")
    return np.column_stack(cols), names


# --------------------------------------------------------- part 1
def all_subsets(D, outcome, log):
    y, N = D["ys"][outcome], D["N"]
    # global model fixes c_hat
    Xg, ng = assemble(D, COVS, True)
    bg, cg, pg, okg = irls(Xg, y, N)
    dfg = len(y) - Xg.shape[1]
    chat = max(pearson_chi2(y, N, pg) / dfg, 1.0)
    log(f"[{outcome}] global model: p={Xg.shape[1]} df={dfg} "
        f"Pearson chi2={pearson_chi2(y, N, pg):.2f} c_hat={chat:.4f} ok={okg}")

    models = []
    n = len(y)
    for k in range(len(COVS) + 1):
        for subset in itertools.combinations(COVS, k):
            ixopts = (False, True) if "renter_share" in subset else (False,)
            for ix in ixopts:
                X, names = assemble(D, list(subset), ix)
                beta, cov, p, ok = irls(X, y, N)
                ll = binom_ll(y, N, p)
                npar = X.shape[1]
                K = npar + 1          # + dispersion
                pen = 2 * K + (2 * K * (K + 1) / (n - K - 1) if n - K - 1 > 0
                               else float("inf"))
                qaicc = -2 * ll / chat + pen
                aicc = -2 * ll + pen
                qbic = -2 * ll / chat + K * math.log(n)
                bic = -2 * ll + K * math.log(n)
                se_q = np.sqrt(np.maximum(np.diag(cov) * chat, 0))
                models.append(dict(
                    subset=list(subset), ix=bool(ix), names=names,
                    beta=beta.tolist(), se=se_q.tolist(), npar=npar, K=K,
                    ll=ll, qaicc=qaicc, aicc=aicc, qbic=qbic, bic=bic,
                    converged=ok,
                    chi2_disp=pearson_chi2(y, N, p) / max(n - npar, 1)))
    log(f"[{outcome}] fitted {len(models)} models, "
        f"{sum(1 for m in models if not m['converged'])} non-converged")

    def weights(key):
        v = np.array([m[key] for m in models])
        d = v - v.min()
        w = np.exp(-0.5 * d)
        return w / w.sum()

    wq = weights("qaicc")
    wb = weights("qbic")
    for m, a, b in zip(models, wq, wb):
        m["w_qaicc"] = float(a)
        m["w_qbic"] = float(b)
        m["d_qaicc"] = float(m["qaicc"] - min(x["qaicc"] for x in models))
    models.sort(key=lambda m: m["qaicc"])

    # inclusion probabilities
    terms = COVS + ["role:renter_share"]
    incl = {}
    for t in terms:
        aw = bw = 0.0
        for m in models:
            inm = (m["ix"] if t == "role:renter_share" else t in m["subset"])
            if inm:
                aw += m["w_qaicc"]
                bw += m["w_qbic"]
        incl[t] = dict(akaike=aw, bma_bic=bw)

    # model-averaged coefficients, zero substitution + Buckland SE
    avg = {}
    for t in terms + ["role"]:
        bbar = 0.0
        for m in models:
            j = m["names"].index(t) if t in m["names"] else None
            bbar += m["w_qaicc"] * (m["beta"][j] if j is not None else 0.0)
        usd = 0.0
        for m in models:
            j = m["names"].index(t) if t in m["names"] else None
            bi = m["beta"][j] if j is not None else 0.0
            vi = (m["se"][j] ** 2) if j is not None else 0.0
            usd += m["w_qaicc"] * math.sqrt(vi + (bi - bbar) ** 2)
        # conditional average (over models containing the term)
        wsum = sum(m["w_qaicc"] for m in models if t in m["names"])
        cond = (sum(m["w_qaicc"] * m["beta"][m["names"].index(t)]
                    for m in models if t in m["names"]) / wsum
                if wsum > 0 else float("nan"))
        avg[t] = dict(avg_full=bbar, se_uncond=usd, avg_cond=cond,
                      w_in=wsum)
    return dict(chat=chat, models=models, incl=incl, avg=avg, terms=terms)


# --------------------------------------------------------- part 2
def permute(D, rows, outcome, log, rng):
    y, N = D["ys"][outcome], D["N"]
    bodies = np.array(D["bodies"])
    jr = COVS.index("renter_share")

    designs = {"FULL": (COVS, True), "MIN": (["renter_share"], True)}
    res = {}

    # (a) renter share permuted across bodies, a body's rows moving together
    ubodies = sorted(set(D["bodies"]))
    bval = {b: D["Z"][bodies == b, jr][0] for b in ubodies}
    bmask = {b: (bodies == b) for b in ubodies}
    for dname, (subset, ix) in designs.items():
        X, names = assemble(D, subset, ix)
        beta, cov, p, ok = irls(X, y, N)
        se_cl, G = cluster_se(X, y, N, p, D["bodies"])
        obs = {t: float(beta[names.index(t)])
               for t in ("renter_share", "role:renter_share", "role")}
        cnt = {t: 0 for t in ("renter_share", "role:renter_share")}
        t0 = time.time()
        vals = np.array([bval[b] for b in ubodies])
        for it in range(NPERM):
            perm = rng.permutation(len(ubodies))
            Zp = D["Z"].copy()
            for b, idx in zip(ubodies, perm):
                Zp[bmask[b], jr] = vals[idx]
            Xp, _ = assemble(D, subset, ix, Zcol=Zp)
            bp = beta_only(Xp, y, N)
            for t in cnt:
                if abs(bp[names.index(t)]) >= abs(obs[t]) - 1e-12:
                    cnt[t] += 1
        log(f"[{outcome}] {dname} renter perm {NPERM} in "
            f"{time.time() - t0:.1f}s")
        res[f"{dname}/renter"] = dict(
            design=dname, scheme="renter_share across bodies",
            n_perm=NPERM,
            terms={t: dict(obs=obs[t], n_ge=cnt[t],
                           p=(1 + cnt[t]) / (1 + NPERM),
                           se_model=float(np.sqrt(cov[names.index(t),
                                                      names.index(t)])),
                           se_cluster=float(se_cl[names.index(t)]))
                   for t in cnt},
            role_obs=obs["role"], converged=ok, n_body_clusters=G)

    # (a2) transit_commute permuted across bodies - the one covariate that
    # survives part 1, so it gets the same null rather than an SE ratio.
    jt = COVS.index("transit_commute")
    tval = {b: D["Z"][bodies == b, jt][0] for b in ubodies}
    tvals = np.array([tval[b] for b in ubodies])
    for dname, (subset, ix) in (("FULL", (COVS, True)),
                                ("MIN_T", (["transit_commute"], False))):
        X, names = assemble(D, subset, ix)
        beta, cov, p, ok = irls(X, y, N)
        se_cl, G = cluster_se(X, y, N, p, D["bodies"])
        jn = names.index("transit_commute")
        obs = float(beta[jn])
        cnt = 0
        t0 = time.time()
        for it in range(NPERM):
            perm = rng.permutation(len(ubodies))
            Zp = D["Z"].copy()
            for b, idx in zip(ubodies, perm):
                Zp[bmask[b], jt] = tvals[idx]
            Xp, _ = assemble(D, subset, ix, Zcol=Zp)
            bp = beta_only(Xp, y, N)
            if abs(bp[jn]) >= abs(obs) - 1e-12:
                cnt += 1
        log(f"[{outcome}] {dname} transit perm {NPERM} in "
            f"{time.time() - t0:.1f}s")
        res[f"{dname}/transit"] = dict(
            design=dname, scheme="transit_commute across bodies",
            n_perm=NPERM,
            terms={"transit_commute": dict(
                obs=obs, n_ge=cnt, p=(1 + cnt) / (1 + NPERM),
                se_model=float(np.sqrt(cov[jn, jn])),
                se_cluster=float(se_cl[jn]))},
            converged=ok, n_body_clusters=G)

    # (b) role permuted within body x route cell (swap the two rows)
    cells = {}
    for i, r in enumerate(rows):
        cells.setdefault((r["body"], r["route"]), []).append(i)
    swappable = [v for v in cells.values() if len(v) == 2]
    log(f"[{outcome}] role permutation: {len(swappable)} swappable cells "
        f"of {len(cells)}")
    for dname, (subset, ix) in designs.items():
        X, names = assemble(D, subset, ix)
        beta, cov, p, ok = irls(X, y, N)
        obs_role = float(beta[names.index("role")])
        obs_ix = float(beta[names.index("role:renter_share")])
        cnt_role = cnt_ix = 0
        t0 = time.time()
        for it in range(NPERM):
            rp = D["role"].copy()
            for pair in swappable:
                if rng.random() < 0.5:
                    a, b = pair
                    rp[a], rp[b] = rp[b], rp[a]
            cols = [np.ones(len(y)), rp]
            for q in D["dummies"]:
                cols.append(X[:, names.index(f"route[{q}]")])
            Xp = [np.column_stack(cols)]
            for c in subset:
                Xp.append(D["Z"][:, COVS.index(c)][:, None])
            if ix:
                Xp.append((rp * D["Z"][:, jr])[:, None])
            Xp = np.column_stack(Xp)
            bp = beta_only(Xp, y, N)
            if abs(bp[names.index("role")]) >= abs(obs_role) - 1e-12:
                cnt_role += 1
            if abs(bp[names.index("role:renter_share")]) >= abs(obs_ix) - 1e-12:
                cnt_ix += 1
        log(f"[{outcome}] {dname} role perm {NPERM} in "
            f"{time.time() - t0:.1f}s")
        res[f"{dname}/role"] = dict(
            design=dname, scheme="role swapped within body x route cell",
            n_perm=NPERM, n_swappable_cells=len(swappable),
            terms={"role": dict(obs=obs_role, n_ge=cnt_role,
                                p=(1 + cnt_role) / (1 + NPERM)),
                   "role:renter_share": dict(obs=obs_ix, n_ge=cnt_ix,
                                             p=(1 + cnt_ix) / (1 + NPERM))},
            converged=ok)
    return res


# --------------------------------------------------------- main
def main():
    logf = open(OUT / "subsets.log", "w")

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

    screen = json.load(open(IN2 / "screen.json"))
    assert screen["kept"] == COVS, f"kept set drift: {screen['kept']}"
    rows = load()
    D = build(rows)
    log(f"rows={len(rows)} bodies={len(set(D['bodies']))} "
        f"routes={D['routes']} kept={COVS}")
    log(f"renter sd={D['sd'][0]:.4f} income sd={D['sd'][1]:.4f}")

    results = {"meta": dict(
        rows=len(rows), bodies=len(set(D["bodies"])),
        cells=len({(r['body'], r['route']) for r in rows}),
        clusters=int(D["N"].sum()), routes=D["routes"], covs=COVS,
        n_perm=NPERM, seed=SEED, route_ref=ROUTE_REF,
        cov_mean=D["mu"].tolist(), cov_sd=D["sd"].tolist(),
        source="output/renter_share_3/outcomes_role_stacked.csv",
        screen="output/renter_share_2/screen.json")}

    rng = np.random.default_rng(SEED)
    for o in OUTCOMES:
        log(f"=== {o} all-subsets")
        results[o] = {"subsets": all_subsets(D, o, log)}
        log(f"=== {o} permutation")
        results[o]["perm"] = permute(D, rows, o, log, rng)

    # ---- csv artefacts
    with open(OUT / "subsets_top.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["outcome", "rank", "terms", "npar", "K", "loglik",
                    "qaicc", "delta_qaicc", "w_qaicc", "qbic", "w_qbic",
                    "aicc", "bic", "beta_renter", "se_renter", "beta_ix",
                    "se_ix", "beta_role", "se_role", "converged"])
        for o in OUTCOMES:
            for i, m in enumerate(results[o]["subsets"]["models"][:25], 1):
                def g(t):
                    if t not in m["names"]:
                        return "", ""
                    j = m["names"].index(t)
                    return f"{m['beta'][j]:+.4f}", f"{m['se'][j]:.4f}"
                br, sr = g("renter_share")
                bi, si = g("role:renter_share")
                bo, so = g("role")
                lab = "+".join([SHORT[c] for c in m["subset"]]) or "(none)"
                if m["ix"]:
                    lab += "+role:renter"
                w.writerow([o, i, lab, m["npar"], m["K"], f"{m['ll']:.3f}",
                            f"{m['qaicc']:.3f}", f"{m['d_qaicc']:.3f}",
                            f"{m['w_qaicc']:.4f}", f"{m['qbic']:.3f}",
                            f"{m['w_qbic']:.4f}", f"{m['aicc']:.3f}",
                            f"{m['bic']:.3f}", br, sr, bi, si, bo, so,
                            int(m["converged"])])

    with open(OUT / "inclusion.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["outcome", "term", "akaike_inclusion_prob",
                    "bic_bma_inclusion_prob", "avg_beta_full",
                    "se_unconditional", "avg_beta_conditional"])
        for o in OUTCOMES:
            s = results[o]["subsets"]
            for t in s["terms"]:
                a = s["avg"][t]
                w.writerow([o, t, f"{s['incl'][t]['akaike']:.4f}",
                            f"{s['incl'][t]['bma_bic']:.4f}",
                            f"{a['avg_full']:+.4f}", f"{a['se_uncond']:.4f}",
                            f"{a['avg_cond']:+.4f}"])
            a = s["avg"]["role"]
            w.writerow([o, "role (always in)", "1.0000", "1.0000",
                        f"{a['avg_full']:+.4f}", f"{a['se_uncond']:.4f}",
                        f"{a['avg_cond']:+.4f}"])

    with open(OUT / "permutation.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["outcome", "design", "scheme", "term", "beta_obs",
                    "n_perm", "n_ge_abs", "p_perm", "se_model", "se_cluster"])
        for o in OUTCOMES:
            for key, r in results[o]["perm"].items():
                for t, v in r["terms"].items():
                    w.writerow([o, r["design"], r["scheme"], t,
                                f"{v['obs']:+.4f}", r["n_perm"], v["n_ge"],
                                f"{v['p']:.4f}",
                                f"{v.get('se_model', float('nan')):.4f}"
                                if "se_model" in v else "",
                                f"{v.get('se_cluster', float('nan')):.4f}"
                                if "se_cluster" in v else ""])

    slim = {"meta": results["meta"]}
    for o in OUTCOMES:
        s = results[o]["subsets"]
        slim[o] = dict(chat=s["chat"], incl=s["incl"], avg=s["avg"],
                       n_models=len(s["models"]),
                       top=[{k: m[k] for k in
                             ("subset", "ix", "npar", "K", "ll", "qaicc",
                              "d_qaicc", "w_qaicc", "qbic", "w_qbic", "aicc",
                              "bic", "names", "beta", "se", "chi2_disp")}
                            for m in s["models"][:25]],
                       perm=results[o]["perm"])
    json.dump(slim, open(OUT / "subsets_results.json", "w"), indent=2)
    log("wrote subsets_top.csv inclusion.csv permutation.csv "
        "subsets_results.json")
    logf.close()


if __name__ == "__main__":
    sys.exit(main())
