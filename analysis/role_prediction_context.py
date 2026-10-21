#!/usr/bin/env python
"""Supplement to role_prediction.py: the floor and the ceiling of the ablation.

Every ablation set in role_prediction.py carries the route and family dummies,
so none of them is a true zero-information baseline. This adds:

  context   route + family dummies only (the floor: route base rates)
  all_noctx everything except route and family (content only)

so a reader can tell how much of each ablation AUC is content and how much is
knowing which pipeline route the turn came from.

Reads the parquet feature tables role_prediction.py wrote. Read-only.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path.home() / "ytpt/analysis"))
from role_prediction import (  # noqa: E402
    CONT, BIN_CORE, FAMILIES, FRAMES, OUT, ROUTES, SUB_TOPICS,
    grouped_cv, mk_gbm, mk_logit,
)

CTX = [f"rt_{r}" for r in ROUTES[1:]] + [f"fam_{f}" for f in FAMILIES[1:]]
CONTENT = CONT + BIN_CORE + [f"fr_{f}" for f in FRAMES] + [f"st_{s}" for s in SUB_TOPICS]
SETS = {"context": CTX, "all_noctx": CONTENT}


def main() -> None:
    turns = pd.read_parquet(OUT / "turns_features.parquet")
    clusters = pd.read_parquet(OUT / "clusters_features.parquet")
    blocks = [
        ("turn_A_cue", turns, "role_cue"),
        ("turn_B_audit", turns, "role_audit"),
        ("cluster_A_cue", clusters, "role_cue"),
        ("cluster_B_audit", clusters, "role_audit_cluster"),
    ]
    rows = []
    for name, df, tgt in blocks:
        y = (df[tgt] == "public").astype(int).to_numpy()
        for fs, cols in SETS.items():
            X = df[cols].to_numpy(float)
            for mname, fn in (("logit_l2", mk_logit()), ("gbm", mk_gbm())):
                for gname in ("body", "episode_id"):
                    res, _ = grouped_cv(X, y, df[gname].to_numpy(), fn)
                    res.update(block=name, target=tgt, feature_set=fs, model=mname,
                               cv=f"leave-{gname}-out", n_features=len(cols))
                    rows.append(res)
        print(f"[ctx] {name} done", flush=True)
    t = pd.DataFrame(rows)[
        ["block", "target", "n", "pos_rate", "feature_set", "model", "cv",
         "n_features", "auc", "log_loss", "base_log_loss", "brier",
         "cal_intercept", "cal_slope"]]
    t.to_csv(OUT / "cv_performance_context.csv", index=False)
    print(t.to_string(), flush=True)

    main_t = pd.read_csv(OUT / "cv_performance.csv")
    pd.concat([main_t, t], ignore_index=True).to_csv(
        OUT / "cv_performance_all.csv", index=False)
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
