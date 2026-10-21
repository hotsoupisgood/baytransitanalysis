#!/usr/bin/env python3
"""RENTER-SHARE-4b — selection-aware permutation null.

The part-1 best model picks renter_share with a large coefficient; the part-2
permutation test is on the FULL and MIN designs, which give a much smaller one.
Quoting the selected coefficient against the FULL design's null would be
post-selection inference. This re-runs the WHOLE all-subsets selection inside
every permutation and nulls the selected quantities:

  stat A  model-averaged beta_renter (Akaike weights over all 384 models)
  stat B  beta_renter in the QAICc-best model
  stat C  Akaike inclusion probability of renter_share
  stat D  model-averaged beta_transit  (same, for transit_commute)
  stat E  Akaike inclusion probability of transit_commute

renter_share and transit_commute are permuted across bodies (every row of a
body moving together), one permutation re-used for both so the two statistics
share a null draw. c_hat is re-estimated from the permuted global model so the
weights are internally consistent in every draw.

EXPLORATORY. No spend, no API, no DB write, nothing committed.
"""
import csv
import itertools
import json
import math
import pathlib
import time

import numpy as np
from scipy.special import gammaln

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "renter_share_4"
COVS = ["renter_share", "log_income", "log_density", "built_2010_plus",
        "nhwhite_share", "transit_commute", "vacancy_rate", "log_population"]
ROUTE_REF = "classifier"
OUTCOMES = ["nimby", "yimby"]
NPERM = 2000
SEED = 20261020
TARGETS = ["renter_share", "transit_commute"]

SUBSETS = []
for k in range(len(COVS) + 1):
    for s in itertools.combinations(COVS, k):
        for ix in ((False, True) if "renter_share" in s else (False,)):
            SUBSETS.append((list(s), ix))


def load():
    rows = []
    for r in csv.DictReader(open(ROOT / "output" / "renter_share_3"
                                 / "outcomes_role_stacked.csv")):
        d = dict(route=r["route"], body=r["body"],
                 role_code=int(r["role_code"]), n=int(r["n"]),
                 nimby=int(r["nimby"]), yimby=int(r["yimby"]))
        for c in COVS:
            d[c] = float(r[c])
        rows.append(d)
    return rows


def beta_p(X, y, N, maxit=120, tol=1e-9):
    beta = np.zeros(X.shape[1])
    for _ in range(maxit):
        eta = np.clip(X @ beta, -30, 30)
        p = 1.0 / (1.0 + np.exp(-eta))
        W = np.maximum(N * p * (1 - p), 1e-9)
        z = eta + (y - N * p) / W
        new = np.linalg.pinv((X.T * W) @ X) @ ((X.T * W) @ z)
        if np.max(np.abs(new - beta)) < tol:
            beta = new
            break
        beta = new
    eta = np.clip(X @ beta, -30, 30)
    return beta, 1.0 / (1.0 + np.exp(-eta))


def main():
    logf = open(OUT / "selperm.log", "w")

    def log(m):
        line = f"[{time.strftime('%H:%M:%S')}] {m}"
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

    rows = load()
    n = len(rows)
    routes = sorted({r["route"] for r in rows})
    dummies = [q for q in routes if q != ROUTE_REF]
    Zraw = np.column_stack([[r[c] for r in rows] for c in COVS]).astype(float)
    Z0 = (Zraw - Zraw.mean(0)) / np.where(Zraw.std(0, ddof=0) > 0,
                                          Zraw.std(0, ddof=0), 1.0)
    role = np.array([r["role_code"] for r in rows], float)
    base = np.column_stack(
        [np.ones(n), role]
        + [np.array([1.0 if r["route"] == q else 0.0 for r in rows])
           for q in dummies])
    N = np.array([r["n"] for r in rows], float)
    bodies = np.array([r["body"] for r in rows])
    ubodies = sorted(set(bodies.tolist()))
    bmask = {b: bodies == b for b in ubodies}
    jr, jt = COVS.index("renter_share"), COVS.index("transit_commute")
    log(f"rows={n} bodies={len(ubodies)} models={len(SUBSETS)} nperm={NPERM}")

    # binomial log-lik constant, vectorised
    def stats(Z, y):
        """All-subsets on design Z; returns the five selected statistics."""
        const = (gammaln(N + 1) - gammaln(y + 1) - gammaln(N - y + 1)).sum()
        cols = {c: Z[:, COVS.index(c)][:, None] for c in COVS}
        ixc = (role * Z[:, jr])[:, None]
        # global model for c_hat
        Xg = np.column_stack([base] + [cols[c] for c in COVS] + [ixc])
        _, pg = beta_p(Xg, y, N)
        pg = np.clip(pg, 1e-10, 1 - 1e-10)
        chat = max(float(np.sum((y - N * pg) ** 2 / (N * pg * (1 - pg))))
                   / (n - Xg.shape[1]), 1.0)
        qa, betas = np.empty(len(SUBSETS)), []
        for i, (s, ix) in enumerate(SUBSETS):
            X = np.column_stack([base] + [cols[c] for c in s]
                                + ([ixc] if ix else []))
            b, p = beta_p(X, y, N)
            p = np.clip(p, 1e-12, 1 - 1e-12)
            ll = const + float(np.sum(y * np.log(p) + (N - y) * np.log1p(-p)))
            K = X.shape[1] + 1
            qa[i] = (-2 * ll / chat + 2 * K
                     + (2 * K * (K + 1) / (n - K - 1) if n - K - 1 > 0
                        else 1e9))
            betas.append(b)
        w = np.exp(-0.5 * (qa - qa.min()))
        w /= w.sum()
        best = int(np.argmin(qa))
        out = {}
        for t, jcov in (("renter_share", jr), ("transit_commute", jt)):
            avg = incl = 0.0
            for i, (s, ix) in enumerate(SUBSETS):
                if t in s:
                    avg += w[i] * betas[i][base.shape[1] + s.index(t)]
                    incl += w[i]
            sb, sx = SUBSETS[best]
            bb = (betas[best][base.shape[1] + sb.index(t)]
                  if t in sb else 0.0)
            out[t] = dict(avg=avg, incl=incl, best=bb)
        out["_best_label"] = "+".join(SUBSETS[best][0]) + (
            "+ix" if SUBSETS[best][1] else "")
        out["_chat"] = chat
        return out

    rng = np.random.default_rng(SEED)
    res = {"meta": dict(n_perm=NPERM, seed=SEED, n_models=len(SUBSETS),
                        rows=n, bodies=len(ubodies), targets=TARGETS,
                        scheme="renter_share and transit_commute permuted "
                               "across bodies, rows of a body together; "
                               "all-subsets selection re-run inside every "
                               "permutation")}
    for o in OUTCOMES:
        y = np.array([r[o] for r in rows], float)
        t0 = time.time()
        obs = stats(Z0, y)
        log(f"[{o}] observed in {time.time() - t0:.1f}s "
            f"best={obs['_best_label']} chat={obs['_chat']:.3f}")
        for t in TARGETS:
            log(f"[{o}]   {t}: avg={obs[t]['avg']:+.4f} "
                f"best={obs[t]['best']:+.4f} incl={obs[t]['incl']:.4f}")
        cnt = {t: dict(avg=0, best=0, incl=0) for t in TARGETS}
        draws = {t: dict(avg=[], best=[], incl=[]) for t in TARGETS}
        t0 = time.time()
        rv = np.array([Z0[bmask[b], jr][0] for b in ubodies])
        tv = np.array([Z0[bmask[b], jt][0] for b in ubodies])
        for it in range(NPERM):
            Zp = Z0.copy()
            pr, pt = rng.permutation(len(ubodies)), rng.permutation(len(ubodies))
            for b, ir, itx in zip(ubodies, pr, pt):
                Zp[bmask[b], jr] = rv[ir]
                Zp[bmask[b], jt] = tv[itx]
            s = stats(Zp, y)
            for t in TARGETS:
                for k in ("avg", "best"):
                    draws[t][k].append(s[t][k])
                    if abs(s[t][k]) >= abs(obs[t][k]) - 1e-12:
                        cnt[t][k] += 1
                draws[t]["incl"].append(s[t]["incl"])
                if s[t]["incl"] >= obs[t]["incl"] - 1e-12:
                    cnt[t]["incl"] += 1
            if (it + 1) % 200 == 0:
                el = time.time() - t0
                log(f"[{o}] perm {it + 1}/{NPERM} {el:.0f}s "
                    f"eta {el / (it + 1) * (NPERM - it - 1):.0f}s")
        res[o] = {t: {k: dict(
            obs=float(obs[t][k]), n_ge=cnt[t][k],
            p=(1 + cnt[t][k]) / (1 + NPERM),
            null_sd=float(np.std(draws[t][k], ddof=1)),
            null_q=[float(q) for q in np.percentile(
                draws[t][k], [2.5, 50, 97.5])])
            for k in ("avg", "best", "incl")} for t in TARGETS}
        res[o]["best_label_obs"] = obs["_best_label"]
        res[o]["chat_obs"] = obs["_chat"]
        log(f"[{o}] done in {time.time() - t0:.0f}s")

    json.dump(res, open(OUT / "selperm_results.json", "w"), indent=2)
    with open(OUT / "selperm.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["outcome", "term", "statistic", "observed", "n_perm",
                    "n_ge", "p_perm", "null_sd", "null_p2.5", "null_median",
                    "null_p97.5"])
        for o in OUTCOMES:
            for t in TARGETS:
                for k, lab in (("avg", "model-averaged beta"),
                               ("best", "beta in QAICc-best model"),
                               ("incl", "Akaike inclusion prob")):
                    v = res[o][t][k]
                    w.writerow([o, t, lab, f"{v['obs']:+.4f}", NPERM,
                                v["n_ge"], f"{v['p']:.4f}",
                                f"{v['null_sd']:.4f}",
                                f"{v['null_q'][0]:+.4f}",
                                f"{v['null_q'][1]:+.4f}",
                                f"{v['null_q'][2]:+.4f}"])
    log("wrote selperm.csv selperm_results.json")
    logf.close()


if __name__ == "__main__":
    main()
