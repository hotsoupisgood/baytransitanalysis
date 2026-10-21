#!/usr/bin/env python
"""Supplement 2 to role_prediction.py: split stance into direction and strength.

The raw `stance` ablation is strong, but among turns with a numeric stance
73 % of dais turns are exactly 0 against 21 % of public turns, while the signed
means barely differ. So "stance" may be carrying *opinionatedness* (a speech
act) rather than *position* (the political content the study is about). This
script separates them:

  stance_dir       signed stance only (direction), numeric turns scored, 0 else
  stance_strength  |stance| + has_numeric_stance (no direction at all)
  speech_act2      strength + length + procedural frame + sub_topic other
  all_nodir        everything, with signed stance replaced by |stance|
  all              the full set, for reference

If stance_strength ~ stance and stance_dir ~ chance, separability is a speech
act and not a position. Also refits the hierarchical logistic with an
identifiable sub_topic coding (st_other as reference), which the first pass
left saturated (hence rhat > 1.01 on the st_* block there).

Reads the parquet feature tables role_prediction.py wrote. Read-only.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path.home() / "ytpt/analysis"))
from role_prediction import (  # noqa: E402
    BIN_CORE, CONT, FAMILIES, FRAMES, OUT, ROUTES, SUB_TOPICS,
    grouped_cv, mk_gbm, mk_logit, pymc_hier,
)

CTX = [f"rt_{r}" for r in ROUTES[1:]] + [f"fam_{f}" for f in FAMILIES[1:]]
FR = [f"fr_{f}" for f in FRAMES]
ST_REF = [f"st_{s}" for s in SUB_TOPICS if s != "other"]   # st_other = reference


def add_decomp(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    d["abs_stance"] = d["stance_median_f"].abs()
    d["stance_pos"] = d["stance_median_f"].clip(lower=0)
    d["stance_neg"] = (-d["stance_median_f"]).clip(lower=0)
    return d


SETS = {
    "stance_dir": ["stance_median_f"] + CTX,
    "stance_strength": ["abs_stance", "has_numeric_stance"] + CTX,
    "stance_dir_and_strength": ["stance_median_f", "abs_stance",
                                "has_numeric_stance"] + CTX,
    "speech_act2": ["abs_stance", "has_numeric_stance", "log_words",
                    "fr_process_notification", "st_other"] + CTX,
    "all_nodir": ["abs_stance", "hostility_median_f", "log_words"] + BIN_CORE
                 + FR + ST_REF + CTX,
    "all": CONT + BIN_CORE + FR + ST_REF + CTX,
}
PYMC_COLS = ["stance_median_f", "abs_stance", "hostility_median_f", "log_words",
             "has_numeric_stance"] + FR + ST_REF + CTX


def main() -> None:
    turns = add_decomp(pd.read_parquet(OUT / "turns_features.parquet"))
    clusters = add_decomp(pd.read_parquet(OUT / "clusters_features.parquet"))
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
                    res.update(block=name, target=tgt, feature_set=fs,
                               model=mname, cv=f"leave-{gname}-out",
                               n_features=len(cols))
                    rows.append(res)
        print(f"[decomp] {name} done", flush=True)
    t = pd.DataFrame(rows)[
        ["block", "target", "n", "pos_rate", "feature_set", "model", "cv",
         "n_features", "auc", "log_loss", "base_log_loss", "brier",
         "cal_intercept", "cal_slope"]]
    t.to_csv(OUT / "cv_performance_stance_decomp.csv", index=False)
    print(t.to_string(), flush=True)

    # identifiable hierarchical refit, with stance split into direction+strength
    for name, df, tgt in blocks:
        print(f"[pymc2] {name}", flush=True)
        y = (df[tgt] == "public").astype(int).to_numpy()
        tab, meta = pymc_hier(df, PYMC_COLS, y, df["body"], label=name,
                              draws=1000, tune=1000)
        tab.to_csv(OUT / f"odds_ratios2_{name}.csv", index=False)
        (OUT / f"odds_ratios2_{name}.json").write_text(json.dumps(meta, indent=2))
        print(tab.head(14).to_string(), flush=True)
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
