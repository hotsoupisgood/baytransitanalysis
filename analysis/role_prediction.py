#!/usr/bin/env python
"""Predict speaker role (dais vs public) from what the turn says.

Two targets, because of circularity:
  A  cue role      = turn_stance.role, from scripts/role_label.py diarization
                     cues. Independent of the stance LLM pass.
  B  audited role  = role_audit.role_audit (turn) / cluster_role_audit
                     (cluster), rule ra-1, which is built FROM the stance
                     model's self_id. self_id is therefore never a feature
                     anywhere in this script (it is not a feature for A either,
                     so the two targets see the same design matrix).

Units: turn (13,123 coded turns) and episode x cluster (2,813 with a stance).
Read-only on state.sqlite. No API, no network, no DB write, nothing committed.

Usage: .venv/bin/python analysis/role_prediction.py [--quick]
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

DB = Path.home() / ".local/share/ytpt/state.sqlite"
OUT = Path.home() / "ytpt/output/role_prediction"
CLUSTER_CSV = Path.home() / "ytpt/output/stance_s1/nimby_view/speaker_clusters_roleaudit.csv"

FRAMES = [
    "traffic_parking", "neighborhood_character", "safety_crime",
    "environment_ceqa_fire", "affordability_displacement", "property_values",
    "process_notification", "fiscal", "state_mandate", "yimby_pro_housing",
    "transit_service", "transit_safety",
]
SUB_TOPICS = [
    "housing_element", "specific_project", "zoning_upzoning", "adu", "parking",
    "affordability_displacement", "rail_service", "bus_service", "bike_ped",
    "road_diet", "transit_funding", "safety", "other",
]
ROUTES = ["classifier", "agenda_index", "local_asr", "swagit_transcript", "fullasr"]
FAMILIES = ["density", "transit", "both"]

CONT = ["stance_median_f", "hostility_median_f", "log_words"]
BIN_CORE = ["has_numeric_stance"]


# ---------------------------------------------------------------- data

def load_turns() -> pd.DataFrame:
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    df = pd.read_sql_query(
        """
        SELECT t.episode_id, t.cluster, t.turn_idx, t.start_s, t.end_s,
               t.words, t.role AS role_cue, t.body, t.family,
               t.stance, t.stance_median, t.hostility_median,
               t.frames, t.sub_topic,
               e.method AS route,
               r.role_audit, r.cluster_role_audit, r.rule
        FROM turn_stance t
        LEFT JOIN episodes e ON e.episode_id = t.episode_id
        LEFT JOIN role_audit r
               ON r.episode_id = t.episode_id AND r.cluster = t.cluster
              AND r.turn_idx = t.turn_idx AND r.prompt_version = t.prompt_version
        """,
        con,
    )
    con.close()
    return df


def featurise_turns(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["has_numeric_stance"] = (d["stance"] != "not_applicable").astype(float)
    d["stance_median_f"] = d["stance_median"].fillna(0.0)
    d["hostility_median_f"] = d["hostility_median"].fillna(0.0)
    d["log_words"] = np.log1p(d["words"].fillna(0).astype(float))
    fr = d["frames"].fillna("")
    for f in FRAMES:
        d[f"fr_{f}"] = fr.str.contains(f, regex=False).astype(float)
    for s in SUB_TOPICS:
        d[f"st_{s}"] = (d["sub_topic"].fillna("other") == s).astype(float)
    for r in ROUTES:
        d[f"rt_{r}"] = (d["route"].fillna("classifier") == r).astype(float)
    for fam in FAMILIES:
        d[f"fam_{fam}"] = (d["family"].fillna("both") == fam).astype(float)
    d["unit"] = d["episode_id"] + "|" + d["cluster"]
    return d


def build_clusters(turns: pd.DataFrame) -> pd.DataFrame:
    """episode x cluster aggregate, restricted to clusters with a stance."""
    cl = pd.read_csv(CLUSTER_CSV)
    cl["unit"] = cl["episode_id"] + "|" + cl["cluster"]
    have_stance = set(cl.loc[cl["label"] != "no stance", "unit"])

    g = turns.groupby("unit")
    agg = pd.DataFrame({
        "episode_id": g["episode_id"].first(),
        "cluster": g["cluster"].first(),
        "body": g["body"].first(),
        "route": g["route"].first(),
        "family": g["family"].first(),
        "n_turns": g.size(),
        "start_s": g["start_s"].min(),
        "end_s": g["end_s"].max(),
        "words": g["words"].sum(),
        "role_cue": g["role_cue"].first(),
        "role_audit_cluster": g["cluster_role_audit"].first(),
        "stance_median_f": g["stance_median"].median(),
        "hostility_median_f": g["hostility_median"].median(),
        "has_numeric_stance": g["has_numeric_stance"].mean(),
    }).reset_index()
    agg["stance_median_f"] = agg["stance_median_f"].fillna(0.0)
    agg["hostility_median_f"] = agg["hostility_median_f"].fillna(0.0)
    agg["log_words"] = np.log1p(agg["words"].astype(float))
    for f in FRAMES:
        agg[f"fr_{f}"] = g[f"fr_{f}"].max().values
    for s in SUB_TOPICS:   # share of turns on that sub_topic
        agg[f"st_{s}"] = g[f"st_{s}"].mean().values
    for r in ROUTES:
        agg[f"rt_{r}"] = (agg["route"].fillna("classifier") == r).astype(float)
    for fam in FAMILIES:
        agg[f"fam_{fam}"] = (agg["family"].fillna("both") == fam).astype(float)
    agg = agg[agg["unit"].isin(have_stance)].reset_index(drop=True)
    return agg


# ---------------------------------------------------------------- feature sets

def feature_cols(kind: str) -> list[str]:
    fr = [f"fr_{f}" for f in FRAMES]
    st = [f"st_{s}" for s in SUB_TOPICS]
    ctx = [f"rt_{r}" for r in ROUTES[1:]] + [f"fam_{f}" for f in FAMILIES[1:]]
    if kind == "all":
        return CONT + BIN_CORE + fr + st + ctx
    if kind == "stance":
        return ["stance_median_f", "has_numeric_stance"] + ctx
    if kind == "hostility":
        return ["hostility_median_f"] + ctx
    if kind == "frames":
        return fr + ctx
    if kind == "length":
        return ["log_words"] + ctx
    if kind == "speech_act":           # length + procedural + no-stance
        return ["log_words", "has_numeric_stance",
                "fr_process_notification", "st_other"] + ctx
    if kind == "no_length":
        return [c for c in CONT if c != "log_words"] + BIN_CORE + fr + st + ctx
    raise ValueError(kind)


FEATURE_SETS = ["all", "stance", "hostility", "frames", "length",
                "speech_act", "no_length"]


# ---------------------------------------------------------------- CV

def grouped_cv(X, y, groups, model_fn, n_splits=5, seed=0):
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss
    oof = np.full(len(y), np.nan)
    gkf = GroupKFold(n_splits=min(n_splits, len(np.unique(groups))))
    for tr, te in gkf.split(X, y, groups):
        if len(np.unique(y[tr])) < 2:
            continue
        m = model_fn()
        m.fit(X[tr], y[tr])
        oof[te] = m.predict_proba(X[te])[:, 1]
    ok = ~np.isnan(oof)
    p = np.clip(oof[ok], 1e-6, 1 - 1e-6)
    yy = y[ok]
    res = {
        "n": int(ok.sum()),
        "pos_rate": float(yy.mean()),
        "auc": float(roc_auc_score(yy, p)),
        "log_loss": float(log_loss(yy, p)),
        "brier": float(brier_score_loss(yy, p)),
        "base_log_loss": float(log_loss(yy, np.full(len(yy), yy.mean()))),
    }
    # calibration slope / intercept on the logit scale (unpenalised logistic)
    from sklearn.linear_model import LogisticRegression
    lg = np.log(p / (1 - p)).reshape(-1, 1)
    try:
        fit = LogisticRegression(penalty=None, max_iter=4000).fit(lg, yy)
        res["cal_intercept"] = float(fit.intercept_[0])
        res["cal_slope"] = float(fit.coef_[0][0])
    except Exception:
        res["cal_intercept"] = res["cal_slope"] = float("nan")
    return res, oof


def mk_logit(C=1.0):
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    return lambda: Pipeline([
        ("s", StandardScaler()),
        ("m", LogisticRegression(penalty="l2", C=C, max_iter=4000)),
    ])


def mk_gbm():
    from sklearn.ensemble import HistGradientBoostingClassifier
    return lambda: HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.06, max_leaf_nodes=15,
        l2_regularization=1.0, early_stopping=False, random_state=0)


def calibration_table(y, p, bins=10):
    ok = ~np.isnan(p)
    d = pd.DataFrame({"y": y[ok], "p": p[ok]})
    d["bin"] = pd.qcut(d["p"], bins, duplicates="drop")
    t = d.groupby("bin", observed=True).agg(
        n=("y", "size"), mean_p=("p", "mean"), obs=("y", "mean")).reset_index()
    t["bin"] = t["bin"].astype(str)
    return t


# ---------------------------------------------------------------- hierarchical

def pymc_hier(df, cols, y, body, draws=600, tune=600, label=""):
    """Hierarchical logistic: body random intercept, standardized fixed effects."""
    import pymc as pm
    Xraw = df[cols].to_numpy(float)
    mu, sd = Xraw.mean(0), Xraw.std(0)
    sd[sd == 0] = 1.0
    X = (Xraw - mu) / sd
    bodies = pd.Categorical(body)
    bi = bodies.codes.astype("int32")
    nb = len(bodies.categories)
    with pm.Model() as m:
        a = pm.Normal("a", 0.0, 2.0)
        s_b = pm.HalfNormal("s_b", 1.0)
        zb = pm.Normal("zb", 0.0, 1.0, shape=nb)
        b = pm.Normal("b", 0.0, 1.0, shape=len(cols))
        eta = a + s_b * zb[bi] + pm.math.dot(X, b)
        pm.Bernoulli("obs", logit_p=eta, observed=y.astype("int8"))
        idata = pm.sample(draws=draws, tune=tune, chains=2, cores=2,
                          target_accept=0.9, progressbar=False,
                          random_seed=11, idata_kwargs={"log_likelihood": False})
    post = idata.posterior["b"].stack(s=("chain", "draw")).to_numpy()  # k x S
    rows = []
    for i, c in enumerate(cols):
        d = post[i]
        rows.append({
            "feature": c, "scale": "per 1 SD" if c in CONT else "per 1 SD (binary)",
            "sd_x": float(sd[i]),
            "beta": float(d.mean()),
            "or": float(np.exp(d.mean())),
            "or_lo90": float(np.exp(np.quantile(d, 0.05))),
            "or_hi90": float(np.exp(np.quantile(d, 0.95))),
            "pr_gt0": float((d > 0).mean()),
        })
    t = pd.DataFrame(rows).sort_values("beta", key=abs, ascending=False)
    sb = idata.posterior["s_b"].to_numpy().ravel()
    meta = {"label": label, "sigma_body_mean": float(sb.mean()),
            "sigma_body_lo90": float(np.quantile(sb, 0.05)),
            "sigma_body_hi90": float(np.quantile(sb, 0.95)),
            "n": int(len(y)), "n_bodies": int(nb)}
    return t, meta


# ---------------------------------------------------------------- main

def run_block(name, df, target_col, group_cols, quick):
    """One (unit x target) block: ablation CV for logit + gbm."""
    y = (df[target_col] == "public").astype(int).to_numpy()
    rows = []
    oof_store = {}
    for fs in FEATURE_SETS:
        cols = feature_cols(fs)
        X = df[cols].to_numpy(float)
        for mname, fn in (("logit_l2", mk_logit()), ("gbm", mk_gbm())):
            if quick and mname == "gbm" and fs not in ("all",):
                continue
            for gname in group_cols:
                g = df[gname].to_numpy()
                res, oof = grouped_cv(X, y, g, fn)
                res.update(block=name, target=target_col, feature_set=fs,
                           model=mname, cv=f"leave-{gname}-out",
                           n_features=len(cols))
                rows.append(res)
                if fs == "all":
                    oof_store[(mname, gname)] = oof
    return pd.DataFrame(rows), y, oof_store


def permutation_importance_gbm(df, cols, y, groups, out_csv, seed=0):
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(seed)
    X = df[cols].to_numpy(float)
    gkf = GroupKFold(n_splits=5)
    imp = {c: [] for c in cols}
    for tr, te in gkf.split(X, y, groups):
        m = mk_gbm()()
        m.fit(X[tr], y[tr])
        base = roc_auc_score(y[te], m.predict_proba(X[te])[:, 1])
        for j, c in enumerate(cols):
            drops = []
            for _ in range(3):
                Xp = X[te].copy()
                Xp[:, j] = rng.permutation(Xp[:, j])
                drops.append(base - roc_auc_score(y[te], m.predict_proba(Xp)[:, 1]))
            imp[c].append(float(np.mean(drops)))
    t = pd.DataFrame({"feature": list(imp), "auc_drop": [np.mean(v) for v in imp.values()],
                      "auc_drop_sd": [np.std(v) for v in imp.values()]})
    t = t.sort_values("auc_drop", ascending=False)
    t.to_csv(out_csv, index=False)
    return t


def hhmmss(x):
    if pd.isna(x):
        return ""
    x = int(x)
    return f"{x // 3600:d}:{(x % 3600) // 60:02d}:{x % 60:02d}" if x >= 3600 \
        else f"{x // 60:02d}:{x % 60:02d}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--no-pymc", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    turns = featurise_turns(load_turns())
    clusters = build_clusters(turns)
    print(f"[data] turns={len(turns)} clusters_with_stance={len(clusters)} "
          f"bodies={turns.body.nunique()} episodes={turns.episode_id.nunique()}",
          flush=True)

    turns.to_parquet(OUT / "turns_features.parquet")
    clusters.to_parquet(OUT / "clusters_features.parquet")

    blocks = [
        ("turn_A_cue", turns, "role_cue", ["body", "episode_id"]),
        ("turn_B_audit", turns, "role_audit", ["body", "episode_id"]),
        ("cluster_A_cue", clusters, "role_cue", ["body", "episode_id"]),
        ("cluster_B_audit", clusters, "role_audit_cluster", ["body", "episode_id"]),
    ]
    all_rows, oofs = [], {}
    for name, df, tgt, gc in blocks:
        print(f"[cv] {name}", flush=True)
        t, y, oof = run_block(name, df, tgt, gc, a.quick)
        all_rows.append(t)
        oofs[name] = (y, oof)
    perf = pd.concat(all_rows, ignore_index=True)
    perf = perf[["block", "target", "unit_n" if "unit_n" in perf else "n",
                 "pos_rate", "feature_set", "model", "cv", "n_features",
                 "auc", "log_loss", "base_log_loss", "brier",
                 "cal_intercept", "cal_slope"]]
    perf.to_csv(OUT / "cv_performance.csv", index=False)
    print(perf.to_string(), flush=True)

    # calibration of the headline model
    for name in oofs:
        y, store = oofs[name]
        for (mname, gname), oof in store.items():
            if gname != "body":
                continue
            calibration_table(y, oof).to_csv(
                OUT / f"calibration_{name}_{mname}.csv", index=False)

    # permutation importance, GBM, all features, leave-body-out
    cols = feature_cols("all")
    for name, df, tgt in (("turn_A_cue", turns, "role_cue"),
                          ("cluster_A_cue", clusters, "role_cue")):
        y = (df[tgt] == "public").astype(int).to_numpy()
        print(f"[perm] {name}", flush=True)
        permutation_importance_gbm(df, cols, y, df["body"].to_numpy(),
                                   OUT / f"perm_importance_{name}.csv")

    # hierarchical logistic, odds ratios
    if not a.no_pymc:
        for name, df, tgt in (("turn_A_cue", turns, "role_cue"),
                              ("turn_B_audit", turns, "role_audit"),
                              ("cluster_A_cue", clusters, "role_cue"),
                              ("cluster_B_audit", clusters, "role_audit_cluster")):
            print(f"[pymc] {name}", flush=True)
            y = (df[tgt] == "public").astype(int).to_numpy()
            t, meta = pymc_hier(df, cols, y, df["body"], label=name,
                                draws=400 if a.quick else 600,
                                tune=400 if a.quick else 600)
            t.to_csv(OUT / f"odds_ratios_{name}.csv", index=False)
            (OUT / f"odds_ratios_{name}.json").write_text(json.dumps(meta, indent=2))
            print(t.head(15).to_string(), flush=True)

    # ---- disagreement with ra-1, cluster level, target A out-of-fold
    y = (clusters["role_cue"] == "public").astype(int).to_numpy()
    X = clusters[cols].to_numpy(float)
    _, oof_l = grouped_cv(X, y, clusters["body"].to_numpy(), mk_logit())
    _, oof_g = grouped_cv(X, y, clusters["body"].to_numpy(), mk_gbm())
    c = clusters.copy()
    c["p_public_logit"] = oof_l
    c["p_public_gbm"] = oof_g
    c["p_public"] = (c["p_public_logit"] + c["p_public_gbm"]) / 2
    c["ra1_flip"] = (c["role_audit_cluster"] != c["role_cue"]).astype(int)
    for thr in (0.5, 0.7, 0.8, 0.9):
        c[f"model_flip_{thr}"] = (
            ((c["role_cue"] == "dais") & (c["p_public"] > thr)) |
            ((c["role_cue"] == "public") & (c["p_public"] < 1 - thr))
        ).astype(int)
    rows = []
    for thr in (0.5, 0.7, 0.8, 0.9):
        mf = c[f"model_flip_{thr}"]
        tp = int(((mf == 1) & (c["ra1_flip"] == 1)).sum())
        fp = int(((mf == 1) & (c["ra1_flip"] == 0)).sum())
        fn = int(((mf == 0) & (c["ra1_flip"] == 1)).sum())
        rows.append({"threshold": thr, "model_flips": int(mf.sum()),
                     "ra1_flips": int(c["ra1_flip"].sum()),
                     "both": tp, "model_only": fp, "ra1_only": fn,
                     "precision_vs_ra1": tp / max(tp + fp, 1),
                     "recall_of_ra1": tp / max(tp + fn, 1)})
    agree = pd.DataFrame(rows)
    agree.to_csv(OUT / "agreement_with_ra1.csv", index=False)
    print(agree.to_string(), flush=True)

    c["conf"] = np.where(c["role_cue"] == "dais", c["p_public"], 1 - c["p_public"])
    c["start"] = c["start_s"].map(hhmmss)
    c["end"] = c["end_s"].map(hhmmss)
    ex = c[c["model_flip_0.8"] == 1].sort_values("conf", ascending=False).head(40)
    keep = ["episode_id", "cluster", "body", "route", "start", "end", "words",
            "n_turns", "role_cue", "role_audit_cluster", "p_public", "conf",
            "ra1_flip", "stance_median_f", "hostility_median_f"]
    ex[keep].to_csv(OUT / "confident_disagreements.csv", index=False)
    c[keep + ["p_public_logit", "p_public_gbm", "model_flip_0.8"]].to_csv(
        OUT / "cluster_predictions.csv", index=False)
    print(ex[keep].head(12).to_string(), flush=True)
    print("[done]", flush=True)


if __name__ == "__main__":
    sys.exit(main())
