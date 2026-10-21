#!/usr/bin/env python3
"""
analysis/stance_s1_models.py -- estimand 2 on the S1 stance sample
(plan/STANCE-S1-RESULTS.md). Reads the scored turns (`turn_stance`, pinned
prompt_version) for the keys in output/stance_s1/s1_turns.csv, joins the
sample's facets (roles_source, locate route), and runs
analysis/models/role_gap.py exactly as the pilot did: cumulative logit, body
and episode random intercepts, family effects, log-words control, locate route
as a fixed-effect stratum, one fit per role instrument (role_source) per
outcome, never pooled across role instruments. Adds 95 % intervals next to the
90 % ones, PPC tables and figures, and the descriptive tables of the pass.

    .venv/bin/python analysis/stance_s1_models.py [--draws 1000 --tune 1000 --chains 4]
"""
import argparse, json, sys, time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "pipeline"))
from analysis.models import role_gap as RG, plots as P  # noqa: E402
import stance, state  # noqa: E402

OUT = ROOT / "output" / "stance_s1"
INSTRUMENT = "stance pass"          # plots.role_gap_* treat anything not starting with STAND-IN as the real pass
SCOPE = "role gap, S1 stratified sample: all public turns + dais turns by body (floor 150), turns >= 5 words"


def load_turns(sample_csv, pv):
    con = state.connect(None, create=False)
    ts = pd.read_sql_query("SELECT episode_id, cluster, turn_idx, stance, stance_median, stance_spread, na_share, hostility_median, "
                           "hostility_spread, n_ok, disagree, words, role, body, family, sub_topic, self_id, frames, contains_instructions, "
                           "cost_usd FROM turn_stance WHERE prompt_version = ?", con, params=(pv,))
    con.close()
    s = pd.read_csv(sample_csv, usecols=["turn_id", "episode_id", "cluster", "turn_idx", "method", "roles_source", "stratum", "pilot", "words"])
    s = s.rename(columns={"words": "words_sample"})
    df = s.merge(ts, on=["episode_id", "cluster", "turn_idx"], how="left", validate="one_to_one")
    df["scored"] = df["stance"].notna()
    df["instrument"] = INSTRUMENT
    df["role_source"] = df["roles_source"]
    return df


def with_95(fit_, table_fn):
    """table_fn at 90 % (module default) plus the same quantities at 95 %."""
    t90 = table_fn(fit_)
    old = RG.CI
    try:
        RG.CI = (2.5, 97.5)
        t95 = table_fn(fit_)
    finally:
        RG.CI = old
    ren = {c: c.replace("90", "95") for c in t95.columns if "90" in c}
    keys = [c for c in ("family", "parameter", "level", "body") if c in t90.columns]
    t95 = t95.rename(columns=ren)[keys + list(ren.values())]
    return t90.merge(t95, on=keys, how="left")


def descriptives(df, out):
    d = df[df["scored"]]
    rows = []
    for (role,), g in d.groupby(["role"]):
        num = g["stance_median"].dropna()
        rows.append({"role": role, "n": len(g), "na_share": round((g["stance"] == "not_applicable").mean(), 3),
                     "stance_mean": round(num.mean(), 3) if len(num) else None,
                     "opposed_share(<=-1)": round((num <= -1).mean(), 3) if len(num) else None,
                     "support_share(>=1)": round((num >= 1).mean(), 3) if len(num) else None,
                     "hostility_mean": round(g["hostility_median"].mean(), 3),
                     "hostile_share(>=2)": round((g["hostility_median"] >= 2).mean(), 3),
                     "k_disagree_share": round(g["disagree"].mean(), 3),
                     "stance_spread_ge2_share": round((g["stance_spread"] >= 2).mean(), 3),
                     "contains_instructions": int(g["contains_instructions"].sum()),
                     "cost_usd": round(g["cost_usd"].sum(), 4)})
    pd.DataFrame(rows).to_csv(out / "s1_descriptives_by_role.csv", index=False)
    by = d.groupby(["body", "role"]).agg(n=("stance", "size"), na_share=("stance", lambda s: (s == "not_applicable").mean()),
                                         stance_mean=("stance_median", "mean"), hostility_mean=("hostility_median", "mean")).reset_index()
    by.to_csv(out / "s1_descriptives_by_body_role.csv", index=False)
    # drift check pilot vs new, same instrument, same k: the only "agreement" available without re-scoring
    drift = d.groupby(["pilot", "role"]).agg(n=("stance", "size"), na_share=("stance", lambda s: (s == "not_applicable").mean()),
                                             stance_mean=("stance_median", "mean"), hostility_mean=("hostility_median", "mean"),
                                             k_disagree=("disagree", "mean"), spread_mean=("stance_spread", "mean")).reset_index()
    drift.to_csv(out / "s1_pilot_vs_new.csv", index=False)
    return rows, drift


def tail_bootstrap(df, out, reps=2000, seed=20261018):
    """Descriptive supplement to the ordinal model: public - dais difference in
    the raw shares opposed (stance <= -1), supportive (>= 1), not_applicable,
    and hostile (hostility >= 2), per role instrument x family, with an
    episode-cluster bootstrap (episodes resampled with replacement inside the
    cell). The proportional-odds PPC shows public stance is bimodal where the
    dais is unimodal at 0, so a single latent shift understates both tails;
    these shares are the model-free statement of the same contrast."""
    rng = np.random.default_rng(seed)
    d = df[df["scored"]].copy()
    d["num"] = d["stance_median"].notna()
    d["opp"] = (d["stance_median"] <= -1) & d["num"]; d["sup"] = (d["stance_median"] >= 1) & d["num"]
    d["na"] = d["stance"] == "not_applicable"; d["host"] = d["hostility_median"] >= 2
    rows = []
    for rs, g0 in d.groupby("role_source"):
        fams = sorted(g0["family"].unique(), key=lambda f: (f != "density", f)) + ["all (turn-weighted)"]
        for fam in fams:
            g = g0 if fam.startswith("all") else g0[g0["family"] == fam]
            eps = g["episode_id"].unique()
            if len(eps) < 5 or g["role"].nunique() < 2:
                continue
            # per-episode counts, so a bootstrap replicate is a weighted sum of rows
            cols = {}
            for role in ("dais", "public"):
                f = g[g["role"] == role]
                agg = f.groupby("episode_id").agg(n=("stance", "size"), num=("num", "sum"), opp=("opp", "sum"), sup=("sup", "sum"),
                                                  na=("na", "sum"), host=("host", "sum")).reindex(eps).fillna(0)
                for c in agg.columns:
                    cols[f"{role}_{c}"] = agg[c].to_numpy(float)
            M = np.column_stack([cols[k] for k in sorted(cols)]); names = sorted(cols)
            ix = {n: i for i, n in enumerate(names)}

            def stats(tot):
                out_ = {}
                for stat, base in (("opp", "num"), ("sup", "num"), ("na", "n"), ("host", "n")):
                    v = {}
                    for role in ("dais", "public"):
                        den = tot[ix[f"{role}_{base}"]]
                        v[role] = tot[ix[f"{role}_{stat}"]] / den if den else np.nan
                    out_[stat] = (v["public"], v["dais"], v["public"] - v["dais"])
                return out_
            obs = stats(M.sum(axis=0))
            boots = {k: [] for k in obs}
            for _ in range(reps):
                w = rng.multinomial(len(eps), np.full(len(eps), 1.0 / len(eps)))
                s = stats(w @ M)
                for k in boots:
                    boots[k].append(s[k][2])
            for k, label in (("opp", "opposed (stance <= -1 | numeric)"), ("sup", "supportive (stance >= 1 | numeric)"),
                             ("na", "not_applicable share"), ("host", "hostile (hostility >= 2)")):
                b = np.array(boots[k], float); b = b[~np.isnan(b)]
                lo90, hi90 = np.percentile(b, [5, 95]) if len(b) else (np.nan, np.nan)
                lo95, hi95 = np.percentile(b, [2.5, 97.5]) if len(b) else (np.nan, np.nan)
                rows.append({"role_source": rs, "family": fam, "statistic": label, "n_turns": int(len(g)), "n_public": int((g["role"] == "public").sum()),
                             "n_episodes": int(len(eps)), "public": obs[k][0], "dais": obs[k][1], "diff_public_minus_dais": obs[k][2],
                             "diff_lo90": lo90, "diff_hi90": hi90, "diff_lo95": lo95, "diff_hi95": hi95, "bootstrap": f"{reps} episode-cluster reps"})
    t = pd.DataFrame(rows)
    t.to_csv(out / "s1_tail_shares_bootstrap.csv", index=False)
    return t


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sample", default=str(OUT / "s1_turns.csv"))
    ap.add_argument("--out", default=str(OUT / "models"))
    ap.add_argument("--draws", type=int, default=1000)
    ap.add_argument("--tune", type=int, default=1000)
    ap.add_argument("--chains", type=int, default=4)
    ap.add_argument("--target-accept", type=float, default=0.9)
    ap.add_argument("--outcomes", default="stance,hostility")
    ap.add_argument("--descriptives-only", action="store_true")
    a = ap.parse_args(argv)
    model = state.setting("SCORING_MODEL", stance.DEFAULT_MODEL)
    pv = stance.prompt_version_for(model)
    out = Path(a.out); tables = out / "tables"; figs = out / "figs"
    tables.mkdir(parents=True, exist_ok=True); figs.mkdir(parents=True, exist_ok=True)
    df = load_turns(a.sample, pv)
    print(f"{len(df)} sampled turns, {int(df['scored'].sum())} scored under {pv}; roles {dict(Counter(df[df['scored']]['role']))}; "
          f"role_source {dict(Counter(df[df['scored']]['role_source']))}", flush=True)
    desc, drift = descriptives(df, out.parent)
    print(pd.DataFrame(desc).to_string(index=False), flush=True)
    tails = tail_bootstrap(df, out.parent)
    print(tails[tails["role_source"] == "db"][["family", "statistic", "n_turns", "public", "dais", "diff_public_minus_dais", "diff_lo90", "diff_hi90"]]
          .round(3).to_string(index=False), flush=True)
    if a.descriptives_only:
        return 0
    turns = df[df["scored"]].copy()
    t0 = time.time()
    fits, skipped = RG.sweep(turns, facet="role_source", outcomes=tuple(a.outcomes.split(",")), stratum="method",
                             draws=a.draws, tune=a.tune, chains=a.chains, target_accept=a.target_accept)
    summary = {"prompt_version": pv, "n_sampled": int(len(df)), "n_scored": int(df["scored"].sum()),
               "settings": {"draws": a.draws, "tune": a.tune, "chains": a.chains, "target_accept": a.target_accept},
               "skipped": {f"{lev}/{oc}": v for (lev, oc), v in skipped.items()}, "fits": {}}
    for (level, outcome), f in fits.items():
        tag = f"{outcome}_{level}"
        gap = with_95(f, RG.gap_table); coef = with_95(f, RG.coef_table); bodies = with_95(f, RG.body_table); ppc = RG.ppc_table(f)
        gap.to_csv(tables / f"role_gap_gap_{tag}.csv", index=False)
        coef.to_csv(tables / f"role_gap_coef_{tag}.csv", index=False)
        bodies.to_csv(tables / f"role_gap_bodies_{tag}.csv", index=False)
        ppc.to_csv(tables / f"role_gap_ppc_{tag}.csv", index=False)
        P.role_gap_forest(gap, figs / f"role_gap_forest_{tag}.png", INSTRUMENT, SCOPE, outcome=outcome,
                          subtitle=f"S1 sample, role instrument {level}, route as stratum; dot = median, bar = 90 %")
        P.role_gap_ppc(ppc, figs / f"role_gap_ppc_{tag}.png", INSTRUMENT, SCOPE, outcome=outcome, levels=f["levels"])
        summary["fits"][tag] = {"n_turns": f["n_turns"], "n_episodes": f["n_episodes"], "strata": f["strata"], "seconds": round(f["seconds"], 1),
                                "diagnostics": f["diagnostics"],
                                "gap": gap[["family", "n_turns", "n_public", "gap_latent_median", "gap_lo90", "gap_hi90", "gap_lo95", "gap_hi95",
                                            "odds_ratio_median", "or_lo90", "or_hi90", "pr_gap_gt_0", "p_tail_dais", "p_tail_public",
                                            "p_tail_diff_median", "p_tail_diff_lo90", "p_tail_diff_hi90"]].to_dict("records")}
        print(f"{tag}: n={f['n_turns']} eps={f['n_episodes']} strata={f['strata']} {f['seconds']:.0f}s diag={f['diagnostics']}", flush=True)
        print(gap[["family", "n_turns", "n_public", "gap_latent_median", "gap_lo90", "gap_hi90", "gap_lo95", "gap_hi95", "pr_gap_gt_0",
                   "p_tail_diff_median", "p_tail_diff_lo90", "p_tail_diff_hi90"]].round(3).to_string(index=False), flush=True)
    summary["seconds"] = round(time.time() - t0, 1)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o)) + "\n")
    print(f"done in {summary['seconds']}s -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
