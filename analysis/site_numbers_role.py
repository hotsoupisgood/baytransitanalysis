#!/usr/bin/env python
"""Recompute every number the site's role panels quote, from the artefacts.

    cd ~/ytpt && .venv/bin/python analysis/site_numbers_role.py

Writes `output/stance_s2/site_numbers_role.json`, which
`~/ytpt-site/export/export_panels.py` and `export_methods.py` read. It exists
for one reason: the dashboard's "Who takes a side" block, its intro paragraph
and its Methods entries quote a dozen numbers that all live in different runs,
and typing any of them into the page by hand is how a corrected number stays
wrong on a website.

Sources, all read-only, nothing recomputed that a run already published:

  output/renter_share_3/models/model_table.csv    delta_role on the ALL-STANCE
                                                  denominator -- the +2.11 that
                                                  exponentiates to the 8.2
  output/renter_share_5/role_restated/            delta_table.csv: the same
                                                  contrast on the DIRECTIONAL
                                                  denominator, and with staff
                                                  clusters dropped
                                                  (plan/REGRESSION-REVIEW-2026-10-19.md
                                                  finding 2)
  output/role_prediction/renter3_role_decomposition.csv
                                                  the two-step split: "took a
                                                  side at all" and "which side"
  output/role_prediction/cv_performance.csv       AUC and calibration slope by
                                                  whole-body holdout
  output/role_prediction/perm_importance_*.csv    which features carry it
  output/renter_share_5/robust/permutation_fl.csv Freedman-Lane and whole-vector
                                                  permutation nulls
  output/stance_s1/nimby_view/speaker_clusters_roleaudit.csv
                                                  the neutral share by role,
                                                  which is what makes the two
                                                  denominators disagree
  ~/.local/share/ytpt/state.sqlite turn_stance    the share of turns sitting at
                                                  exactly stance 0, by role

No YouTube request, no GPU, no Claude call, no API key, $0. No DB write -- the
state DB is opened read-only. `.env`, `plan/frame.csv` and `plan/DECISIONS.md`
are not read or touched. Nothing here chooses a number: where a run published
two denominators, both are written out and the page is what decides which it
leads with.
"""
from __future__ import annotations

import csv
import json
import math
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "output"
DEST = OUT / "stance_s2" / "site_numbers_role.json"
STATE = Path("~/.local/share/ytpt/state.sqlite").expanduser()

RS3_TABLE = OUT / "renter_share_3" / "models" / "model_table.csv"
RESTATED = OUT / "renter_share_5" / "role_restated" / "delta_table.csv"
DECOMP = OUT / "role_prediction" / "renter3_role_decomposition.csv"
CVPERF = OUT / "role_prediction" / "cv_performance.csv"
#: the direction-vs-strength ablation lives in its own sweep, with `stance_dir`
#: (signed stance and nothing else) as its first row
CVDECOMP = OUT / "role_prediction" / "cv_performance_stance_decomp.csv"
PERMFL = OUT / "renter_share_5" / "robust" / "permutation_fl.csv"
CLUSTERS = OUT / "stance_s1" / "nimby_view" / "speaker_clusters_roleaudit.csv"
PERM_IMP = OUT / "role_prediction" / "perm_importance_turn_A_cue.csv"
PERM_IMP_CL = OUT / "role_prediction" / "perm_importance_cluster_A_cue.csv"

#: the labels the nimby_view assigns; "no stance" means no turn carried a number
DIRECTIONAL = ("YIMBY", "NIMBY")
STANCE_BEARING = ("YIMBY", "NIMBY", "neutral")


def rd(path):
    if not Path(path).exists():
        return []
    return list(csv.DictReader(Path(path).open(encoding="utf-8")))


def f(v):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return None if (math.isnan(x) or math.isinf(x)) else x


def role_gap():
    """The same public-vs-dais contrast on both denominators, side by side.

    The all-stance row is run 3's published fit, kept because it is the number
    the owner asked for; the directional rows are run 5's restatement after the
    review. Both are written: the page carries the 8.2 as the headline and the
    split underneath, and neither is reconstructed from the other.
    """
    out = {}
    # -- all clusters with a stance (run 3) ----------------------------------
    # Both published specs are written out. They differ only in which role
    # interactions are in the design, they land two hundredths of a log-odds
    # apart, and they exponentiate to 8.2 (R1) and 8.3 (R2) -- which is why the
    # same finding has been quoted as both. R2 is run 3's own §3a headline and
    # is what `all_stance` points at; R1 stays beside it rather than being
    # quietly dropped.
    for spec in ("R1", "R2"):
        for r in rd(RS3_TABLE):
            if r["outcome"] == "nimby" and r["spec"] == spec and r["paired_only"] == "0" \
                    and r["parameter"] == "delta_role":
                d = f(r["median"])
                out[f"all_stance_{spec.lower()}"] = {
                    "spec": spec,
                    "denominator": "NIMBY out of all clusters with a stance "
                                   "(YIMBY, NIMBY or neutral)",
                    "delta": d, "ci95": [f(r["ci95_lo"]), f(r["ci95_hi"])],
                    "ci90": [f(r["ci90_lo"]), f(r["ci90_hi"])],
                    "odds_ratio": math.exp(d),
                    "or_ci95": [math.exp(f(r["ci95_lo"])), math.exp(f(r["ci95_hi"]))],
                    "or_ci90": [math.exp(f(r["ci90_lo"])), math.exp(f(r["ci90_hi"]))],
                    "n_rows": int(f(r["n_rows"])), "n_bodies": int(f(r["n_bodies"])),
                    "n_clusters": int(f(r["n_clusters"])),
                    "source": f"output/renter_share_3/models/model_table.csv "
                              f"(nimby, {spec}, delta_role)",
                }
                break
    if out.get("all_stance_r2"):
        out["all_stance"] = dict(out["all_stance_r2"])
        out["all_stance"]["spec_note"] = (
            "Run 3's §3a headline spec. The other published spec (R1) gives "
            f"{out['all_stance_r1']['delta']:+.3f}, odds ratio "
            f"{out['all_stance_r1']['odds_ratio']:.2f}."
            if out.get("all_stance_r1") else "")
    # -- directional, and directional without staff (run 5) ------------------
    key = {"DIRECTIONAL": "directional", "DIR_NOSTAFF": "directional_nostaff",
           "ELECT_RESID": "elected_vs_resident"}
    for r in rd(RESTATED):
        k = key.get(r["variant"])
        if not k:
            continue
        out[k] = {
            "denominator": r["label"],
            "delta": f(r["delta_median"]),
            "ci90": [f(r["ci90_lo"]), f(r["ci90_hi"])],
            "ci95": [f(r["ci95_lo"]), f(r["ci95_hi"])],
            "odds_ratio": f(r["or_median"]),
            "or_ci90": [f(r["or_ci90_lo"]), f(r["or_ci90_hi"])],
            "or_ci95": [f(r["or_ci95_lo"]), f(r["or_ci95_hi"])],
            "raw_dais_share": f(r["raw_grp0_share"]),
            "raw_public_share": f(r["raw_grp1_share"]),
            "n_rows": int(f(r["n_rows"])), "n_bodies": int(f(r["n_bodies"])),
            "n_dir_clusters": int(f(r["n_dir_clusters"])),
            "source": "output/renter_share_5/role_restated/delta_table.csv",
        }
    return out


def decomposition():
    """"Took a side at all" and "which side, given a side", audited role."""
    want = {"opinionated|has_stance role=ra1_audit": "took_a_side",
            "NIMBY|opinionated role=ra1_audit": "which_side_nimby",
            "NIMBY|has_stance role=ra1_audit": "nimby_of_stance_unadjusted"}
    out = {}
    for r in rd(DECOMP):
        k = want.get(r["model"])
        if not k:
            continue
        out[k] = {
            "beta": f(r["beta_role"]),
            "ci90": [f(r["lo90"]), f(r["hi90"])],
            "odds_ratio": f(r["or_role"]),
            "n": int(f(r["n"])), "n_public": int(f(r["n_public"])),
            "rate_dais": f(r["pos_rate_dais"]), "rate_public": f(r["pos_rate_public"]),
            "source": "output/role_prediction/renter3_role_decomposition.csv",
        }
    out["note"] = ("The two steps are not additive in log-odds -- each is its own "
                   "logistic fit on its own denominator, which is the point: the "
                   "first asks who speaks to a side, the second asks which side.")
    return out


def role_prediction():
    """Can the words alone tell a dais speaker from a public one?

    Only the whole-body holdout is read here: leave-episode-out is optimistic
    for this question, because the same meeting's dais sits in both folds. The
    audited-role block (target B) is written with its circularity flag, because
    the audit is itself fitted on self-identification text.
    """
    rows = rd(CVPERF) + rd(CVDECOMP)

    def pick(block, feature_set, model):
        for r in rows:
            if r["block"] == block and r["feature_set"] == feature_set \
                    and r["model"] == model and r["cv"] == "leave-body-out":
                return {"auc": f(r["auc"]), "cal_slope": f(r["cal_slope"]),
                        "n": int(f(r["n"])), "pos_rate": f(r["pos_rate"]),
                        "n_features": int(f(r["n_features"]))}
        return None

    def drop(path, n=3):
        rs = rd(path)
        if not rs:
            return []
        col = "auc_drop" if "auc_drop" in rs[0] else next(
            (c for c in rs[0] if "drop" in c or "import" in c), None)
        if not col:
            return []
        rs = [r for r in rs if f(r[col]) is not None]
        rs.sort(key=lambda r: -f(r[col]))
        feat = "feature" if "feature" in rs[0] else list(rs[0])[0]
        return [{"feature": r[feat], "auc_drop": round(f(r[col]), 4)} for r in rs[:n]]

    return {
        "cue_role": {
            "cluster": pick("cluster_A_cue", "all", "gbm"),
            "turn": pick("turn_A_cue", "all", "gbm"),
            "cluster_logit": pick("cluster_A_cue", "all", "logit_l2"),
            "signed_stance_only_cluster_logit": pick("cluster_A_cue", "stance_dir", "logit_l2"),
            "signed_stance_only_turn_logit": pick("turn_A_cue", "stance_dir", "logit_l2"),
            "abs_stance_only_cluster_logit": pick("cluster_A_cue", "stance_strength", "logit_l2"),
        },
        "audited_role": {
            "cluster": pick("cluster_B_audit", "all", "gbm"),
            "turn": pick("turn_B_audit", "all", "gbm"),
            "circular": True,
            "circularity_note": ("The audited role is itself derived from what the "
                                 "speaker says about who they are, so a content model "
                                 "predicting it is partly predicting its own input. "
                                 "These AUCs are not comparable to the cue-role ones."),
        },
        "importance_turn": drop(PERM_IMP),
        "importance_cluster": drop(PERM_IMP_CL),
        "holdout": "leave-one-body-out (every meeting of a body held out together)",
        "source": "output/role_prediction/cv_performance.csv, perm_importance_*.csv",
    }


def permutation():
    """The valid partial-coefficient nulls, 5,000 draws each.

    Freedman-Lane and whole-vector body permutation. The role permutation that
    run 4 reported is deliberately absent and flagged withdrawn: see
    plan/RENTER-SHARE-4-SUBSETS.md section 0 -- "No permutation result for role
    should be quoted."
    """
    out = {"n_perm": None, "floor": None, "rows": [],
           "role_withdrawn": {
               "withdrawn": True,
               "was": 0.0002,
               "why": ("A within-cell role swap keeps the row's own counts attached to "
                       "the row, so it tests where the label lands rather than whether "
                       "the gap is zero; and an across-body role permutation is the "
                       "identity here, because every fitted body has the same role "
                       "composition. The role gap rests on the model and the by-body "
                       "cluster-robust standard error, not on a permutation."),
               "source": "plan/RENTER-SHARE-4-SUBSETS.md §0, "
                         "plan/REGRESSION-REVIEW-2026-10-19.md finding 10"}}
    for r in rd(PERMFL):
        n = int(f(r["n_perm"]) or 0)
        out["n_perm"] = n
        out["floor"] = round(1.0 / (1 + n), 6) if n else None
        out["rows"].append({
            "term": r["term"], "mode": r["mode"], "outcome": r["outcome"],
            "design": r["design"], "run": r["run"],
            "beta": f(r["beta_obs"]), "p": f(r["p_two_sided"]),
        })
    # the two the page quotes, picked by name rather than by row order
    def p_of(term, outcome, design_has, mode):
        for r in out["rows"]:
            if r["term"] == term and r["outcome"] == outcome and r["mode"] == mode \
                    and design_has in r["design"]:
                return r["p"]
        return None
    out["quoted"] = {
        "transit_commute_yimby": {"whole_vector": p_of("transit_commute", "YIMBY", "FULL", "whole_vector"),
                                  "freedman_lane": p_of("transit_commute", "YIMBY", "FULL", "freedman_lane")},
        "transit_commute_nimby": {"whole_vector": p_of("transit_commute", "NIMBY", "FULL", "whole_vector"),
                                  "freedman_lane": p_of("transit_commute", "NIMBY", "FULL", "freedman_lane")},
        "renter_share_yimby_full": {"whole_vector": p_of("renter_share", "YIMBY", "FULL", "whole_vector"),
                                    "freedman_lane": p_of("renter_share", "YIMBY", "FULL", "freedman_lane")},
        "renter_share_nimby_full": {"whole_vector": p_of("renter_share", "NIMBY", "FULL", "whole_vector"),
                                    "freedman_lane": p_of("renter_share", "NIMBY", "FULL", "freedman_lane")},
        "renter_share_public_yimby_base": {
            "whole_vector": p_of("renter_share", "public YIMBY", "(a) base", "whole_vector"),
            "freedman_lane": p_of("renter_share", "public YIMBY", "(a) base", "freedman_lane")},
    }
    out["source"] = "output/renter_share_5/robust/permutation_fl.csv"
    return out


def neutral_by_role(gap):
    """Why the two denominators disagree: neutral is a dais phenomenon.

    `gap` is role_gap()'s output, used only so the staff note quotes the two
    fits' own odds ratios and row counts rather than restating them.
    """
    n = defaultdict(lambda: defaultdict(int))
    for r in rd(CLUSTERS):
        n[r["role"]][r["label"]] += 1
    out = {}
    for role, c in n.items():
        bearing = sum(c[k] for k in STANCE_BEARING)
        direct = sum(c[k] for k in DIRECTIONAL)
        out[role] = {
            "clusters_with_stance": bearing,
            "directional": direct,
            "neutral": c["neutral"],
            "neutral_share_of_stance_bearing": round(c["neutral"] / bearing, 4) if bearing else None,
            "nimby": c["NIMBY"], "yimby": c["YIMBY"],
            "nimby_share_of_stance_bearing": round(c["NIMBY"] / bearing, 4) if bearing else None,
            "nimby_share_of_directional": round(c["NIMBY"] / direct, 4) if direct else None,
        }
    out["source"] = "output/stance_s1/nimby_view/speaker_clusters_roleaudit.csv"
    a, b = gap.get("directional") or {}, gap.get("directional_nostaff") or {}
    if a and b:
        out["staff_note"] = (
            "Staff clusters are folded into the dais by the role audit (rule ra-1) "
            "and are almost all neutral. Dropping them is the directional_nostaff "
            f"fit: {a['n_dir_clusters'] - b['n_dir_clusters']} fewer directional "
            f"clusters, and the odds ratio moves {a['odds_ratio']:.2f} → "
            f"{b['odds_ratio']:.2f}.")
    return out


def zero_stance():
    """Share of scored turns sitting at exactly stance 0, by role.

    This is the raw version of the whole finding: neutral is something the dais
    does, not a position the public also holds. Read from the live state DB,
    read-only, at the one prompt_version turn_stance carries.
    """
    if not STATE.exists():
        return {"available": False, "why": f"no state DB at the expected path"}
    con = sqlite3.connect(f"file:{STATE}?mode=ro", uri=True)
    try:
        rows = con.execute(
            "select role, count(*) n, sum(case when stance_median = 0 then 1 else 0 end) at_zero "
            "from turn_stance where stance_median is not null group by role").fetchall()
        total, pv = con.execute(
            "select count(*), min(prompt_version) from turn_stance").fetchone()
    finally:
        con.close()
    out = {"available": True, "turns_total": total, "prompt_version": pv,
           "by_role": {}, "source": "~/.local/share/ytpt/state.sqlite turn_stance",
           "note": ("Denominator is turns carrying a numeric stance, not all scored "
                    "turns: a turn with no number has no position to sit at zero.")}
    for role, n, z in rows:
        out["by_role"][role or "unknown"] = {
            "scored_turns": n, "at_zero": z,
            "share_at_zero": round(z / n, 4) if n else None}
    return out


def main():
    missing = [str(p.relative_to(ROOT)) for p in
               (RS3_TABLE, RESTATED, DECOMP, CVPERF, CVDECOMP, PERMFL, CLUSTERS)
               if not p.exists()]
    gap = role_gap()
    doc = {
        "generated_by": "analysis/site_numbers_role.py",
        "stamp": ("EXPLORATORY — the stance κ ≥ 0.6 hand-coding gate is not passed, and "
                  "role is an AUDIT (ra-1) whose self-identification is another model's "
                  "output that no human has checked."),
        "missing_sources": missing,
        "role_gap": gap,
        "decomposition": decomposition(),
        "neutral_by_role": neutral_by_role(gap),
        "zero_stance": zero_stance(),
        "role_prediction": role_prediction(),
        "permutation": permutation(),
    }
    DEST.parent.mkdir(parents=True, exist_ok=True)
    DEST.write_text(json.dumps(doc, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"[site_numbers_role] wrote {DEST.relative_to(ROOT)} "
          f"({DEST.stat().st_size/1024:.1f} kB)")
    if missing:
        print("[site_numbers_role] MISSING: " + ", ".join(missing))
    g = doc["role_gap"]
    for k in ("all_stance", "directional", "directional_nostaff"):
        if g.get(k):
            v = g[k]
            print(f"   {k:<22} delta {v['delta']:+.3f}  OR {v['odds_ratio']:.2f} "
                  f"(95 % {v.get('or_ci95', [None, None])[0]:.2f}–{v.get('or_ci95', [None, None])[1]:.2f})")
    z = doc["zero_stance"].get("by_role", {})
    for r in sorted(z):
        print(f"   at stance 0: {r:<8} {z[r]['share_at_zero']:.3f} "
              f"({z[r]['at_zero']}/{z[r]['scored_turns']})")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
