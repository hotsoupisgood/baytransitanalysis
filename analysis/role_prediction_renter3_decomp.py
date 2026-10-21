#!/usr/bin/env python
"""Supplement 3: split RENTER-SHARE-3's +2.11 role effect into two questions.

RENTER-SHARE-3-ROLE fits P(NIMBY) over clusters that "have a stance"
(label != "no stance"), and gets role = +2.11 log-odds public vs dais, with
+1.03 on YIMBY *in the same direction*. Both rising together is the signature
of a speech act, not a position: the NIMBY/YIMBY/neutral label is a threshold
(T = 0.5) on mean cluster stance, so a role difference in how far from zero a
cluster's stance sits raises the NIMBY and the YIMBY share at once.

This re-asks the role question as two nested ones on the same rows and the
same unit:

  (1) opinionated   P(label in {NIMBY, YIMBY} | has a stance) ~ role
  (2) direction     P(NIMBY | label in {NIMBY, YIMBY})        ~ role

(1) is a speech act: did this speaker take a side at all. (2) is the political
content the study is about: which side. If the role effect lives in (1) and
not (2), the +2.11 is mostly a speech-act difference.

Body random intercept, route fixed stratum (reference classifier), as in
RENTER-SHARE-3. Both role columns (cue and ra-1 audit) are run. No covariates:
this is a decomposition of the role term, not a re-fit of run 3's design.

Reads output/stance_s1/nimby_view/speaker_clusters_roleaudit.csv and the
role_prediction parquet (for the ra-1 cluster role and the route). Read-only.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pymc as pm

OUT = Path.home() / "ytpt/output/role_prediction"
CSV = Path.home() / "ytpt/output/stance_s1/nimby_view/speaker_clusters_roleaudit.csv"
ROUTES = ["classifier", "agenda_index", "local_asr", "swagit_transcript", "fullasr"]


def fit(df: pd.DataFrame, y: np.ndarray, role_pub: np.ndarray, label: str):
    bodies = pd.Categorical(df["body"])
    bi, nb = bodies.codes.astype("int32"), len(bodies.categories)
    R = np.column_stack([(df["route"] == r).astype(float) for r in ROUTES[1:]])
    with pm.Model():
        a = pm.Normal("a", 0.0, 2.5)
        s_b = pm.HalfNormal("s_b", 1.5)
        zb = pm.Normal("zb", 0.0, 1.0, shape=nb)
        b_role = pm.Normal("b_role", 0.0, 2.5)
        b_rt = pm.Normal("b_rt", 0.0, 1.5, shape=R.shape[1])
        eta = a + s_b * zb[bi] + b_role * role_pub + pm.math.dot(R, b_rt)
        pm.Bernoulli("obs", logit_p=eta, observed=y.astype("int8"))
        idata = pm.sample(1000, tune=1000, chains=2, cores=2, target_accept=0.9,
                          progressbar=False, random_seed=7,
                          idata_kwargs={"log_likelihood": False})
    d = idata.posterior["b_role"].to_numpy().ravel()
    return {
        "model": label, "n": int(len(y)), "n_public": int(role_pub.sum()),
        "pos_rate": float(y.mean()),
        "pos_rate_dais": float(y[role_pub == 0].mean()),
        "pos_rate_public": float(y[role_pub == 1].mean()),
        "beta_role": float(d.mean()),
        "lo90": float(np.quantile(d, 0.05)), "hi90": float(np.quantile(d, 0.95)),
        "lo95": float(np.quantile(d, 0.025)), "hi95": float(np.quantile(d, 0.975)),
        "or_role": float(np.exp(d.mean())), "pr_gt0": float((d > 0).mean()),
    }


def main() -> None:
    cl = pd.read_csv(CSV)
    cl["unit"] = cl["episode_id"] + "|" + cl["cluster"]
    # NB: the view CSV's own `role` column is ALREADY the ra-1 audited role
    # (it is the _roleaudit view). The cue role has to come from the DB.
    feat = pd.read_parquet(OUT / "clusters_features.parquet")[
        ["unit", "route", "role_cue", "role_audit_cluster"]]
    d = cl.merge(feat, on="unit", how="inner")
    d = d[d["label"] != "no stance"].reset_index(drop=True)
    d["route"] = d["route"].fillna("classifier")
    print(f"[rows] {len(d)} clusters with a stance, {d.body.nunique()} bodies",
          flush=True)

    rows = []
    for rc, rl in (("role_cue", "cue"), ("role_audit_cluster", "ra1_audit")):
        pub = (d[rc] == "public").to_numpy().astype(float)
        # (1) opinionated vs neutral, all clusters with a stance
        y1 = d["label"].isin(["NIMBY", "YIMBY"]).to_numpy().astype(int)
        rows.append(fit(d, y1, pub, f"opinionated|has_stance role={rl}"))
        # (2) NIMBY vs YIMBY, opinionated clusters only
        m = y1 == 1
        rows.append(fit(d[m].reset_index(drop=True),
                        (d.loc[m, "label"] == "NIMBY").to_numpy().astype(int),
                        pub[m], f"NIMBY|opinionated role={rl}"))
        # reference: run 3's own two outcomes, same rows, role term only
        for lab in ("NIMBY", "YIMBY"):
            rows.append(fit(d, (d["label"] == lab).to_numpy().astype(int), pub,
                            f"{lab}|has_stance role={rl}"))
        print(f"[fit] {rl} done", flush=True)

    t = pd.DataFrame(rows)
    t.to_csv(OUT / "renter3_role_decomposition.csv", index=False)
    print(t.to_string(), flush=True)

    # plain cross-tab for the report
    for rc, rl in (("role_cue", "cue"), ("role_audit_cluster", "ra1_audit")):
        ct = pd.crosstab(d[rc], d["label"], normalize="index").round(4)
        ct["n"] = pd.crosstab(d[rc], d["label"]).sum(axis=1)
        ct.to_csv(OUT / f"label_by_role_{rl}.csv")
        print(f"\n{rl}\n{ct.to_string()}", flush=True)
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
