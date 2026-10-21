#!/usr/bin/env python3
"""
scripts/role_audit.py -- audit the diarization-cue speaker role
(`turn_stance.role`, from scripts/role_label.py) against what the speaker says
about who they are (`turn_stance.self_id`, the codebook's SELF_ID field), and
write a corrected role to a NEW table. Nothing existing is overwritten.

WHY
---
role_label.py is a text+timing cue labeller; its own docstring names the known
miss: "a councilmember who says 'I live in ...' or 'on behalf of my community'
reads as public". On Petaluma (Swagit route) two of the strongest "public"
clusters are, by the stance model's own rationales, elected members. Dais
speech in the public bucket inflates whatever the public bucket is supposed to
measure. SELF_ID is an independent read of the same turn: it comes from the
turn's words only and never from the diarizer, the timing or the cues.

RULE (audit version ra-1, 2026-10-18)
  self_id in {elected, staff}                            -> role_audit = dais
  self_id in {resident, homeowner, renter, business,
              org_rep}                                   -> role_audit = public
  self_id = unknown                                      -> keep the cue role
A turn's rationale matching /elected|council ?member|councilmember|mayor|
supervisor|director/ while role = 'public' is recorded as `rationale_flag` but
does NOT drive the rule: the rationale is free text about the item, and
"the director said" is someone being described, not the speaker.

Cluster roll-up (the analysis unit is the episode x cluster, DECISIONS #15):
`cluster_role_audit` = majority of that cluster's informative turns, ties and
clusters with no informative turn fall back to the cue role.

OUTPUT
  state.sqlite table `role_audit` (created if absent, rows replaced for the
  audited prompt_version): one row per turn with both roles, the self_id, the
  rule branch taken, the rationale flag and the cluster roll-up.
  output/role_audit/disagreement_{overall,by_route,by_body}.csv
  output/role_audit/flagged_public_turns.csv

RUN
  .venv/bin/python scripts/role_audit.py            # measure + write
  .venv/bin/python scripts/role_audit.py --dry-run  # measure only
"""
import argparse, re, sqlite3, sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "pipeline"))
import state  # noqa: E402

AUDIT_VERSION = "ra-1"
DAIS_SELF = ("elected", "staff")
PUBLIC_SELF = ("resident", "homeowner", "renter", "business", "org_rep")
RATIONALE_RX = re.compile(r"elected|council ?member|councilmember|mayor|supervisor|director", re.I)
OUT = ROOT / "output" / "role_audit"
#: the frame CSV supplies `method` (locate route) and `roles_source` per turn.
#: S1 wrote s1_frame.csv; the S2 top-up (scripts/stance_s2_sample.py) writes
#: frame_all.csv = s1_frame + s2_frame deduped, so the audit covers the turns
#: S2 added as well. The combined file is preferred when it exists; --frame
#: overrides. The rule itself (ra-1) is unchanged.
S1_FRAME = ROOT / "output" / "stance_s1" / "s1_frame.csv"
FRAME_ALL = ROOT / "output" / "stance_s2" / "frame_all.csv"
FRAME = FRAME_ALL if FRAME_ALL.exists() else S1_FRAME

DDL = """
CREATE TABLE IF NOT EXISTS role_audit (
  episode_id TEXT NOT NULL, cluster TEXT NOT NULL, turn_idx INTEGER NOT NULL,
  prompt_version TEXT NOT NULL,
  role_cue TEXT,              -- turn_stance.role, scripts/role_label.py cues (never modified)
  self_id TEXT,               -- turn_stance.self_id, codebook SELF_ID
  role_audit TEXT,            -- dais | public, this file's rule
  rule TEXT,                  -- self_id_dais | self_id_public | keep_cue
  rationale_flag INTEGER,     -- rationale names an elected office while role_cue = public
  cluster_role_audit TEXT,    -- majority roll-up over the episode x cluster
  cluster_rule TEXT,          -- majority | tie_keep_cue | no_signal_keep_cue
  audit_version TEXT NOT NULL, audited_at TEXT NOT NULL,
  PRIMARY KEY (episode_id, cluster, turn_idx, prompt_version))
"""


def load(frame=None):
    con = state.connect(None, create=False)
    ts = pd.read_sql_query(
        "SELECT episode_id, cluster, turn_idx, prompt_version, role, self_id, rationale, body, family, "
        "stance_median, words, start_s, end_s FROM turn_stance", con)
    con.close()
    fr = pd.read_csv(frame or FRAME, usecols=["episode_id", "cluster", "turn_idx", "method", "roles_source"])
    d = ts.merge(fr, on=["episode_id", "cluster", "turn_idx"], how="left", validate="one_to_one")
    miss = int(d.method.isna().sum())
    print(f"loaded {len(ts)} turn_stance rows against {len(fr)} frame rows ({frame or FRAME}); "
          f"{miss} with no frame row (route/roles_source blank)")
    return d


def apply_rule(df):
    d = df.copy()
    d["rule"] = "keep_cue"
    d.loc[d.self_id.isin(DAIS_SELF), "rule"] = "self_id_dais"
    d.loc[d.self_id.isin(PUBLIC_SELF), "rule"] = "self_id_public"
    d["role_audit"] = d["role"]
    d.loc[d.rule == "self_id_dais", "role_audit"] = "dais"
    d.loc[d.rule == "self_id_public", "role_audit"] = "public"
    d["rationale_flag"] = ((d.role == "public") & d.rationale.fillna("").str.contains(RATIONALE_RX)).astype(int)
    # cluster roll-up over informative turns
    inf = d[d.rule != "keep_cue"]
    votes = inf.groupby(["episode_id", "cluster", "role_audit"]).size().unstack(fill_value=0)
    roll = {}
    for key, row in votes.iterrows():
        nd, np_ = int(row.get("dais", 0)), int(row.get("public", 0))
        roll[key] = ("dais", "majority") if nd > np_ else ("public", "majority") if np_ > nd else (None, "tie_keep_cue")
    def pick(r):
        v = roll.get((r.episode_id, r.cluster))
        if v is None:
            return pd.Series([r.role, "no_signal_keep_cue"])
        return pd.Series([v[0] or r.role, v[1]])
    d[["cluster_role_audit", "cluster_rule"]] = d.apply(pick, axis=1)
    return d


def report(d):
    OUT.mkdir(parents=True, exist_ok=True)
    inf = d[d.rule != "keep_cue"].copy()
    inf["disagree"] = (inf.role != inf.role_audit).astype(int)

    def agg(g):
        return pd.Series({"n_turns": len(g), "n_informative": int(len(g[g.rule != "keep_cue"])),
                          "disagree": int((g.role != g.role_audit).sum()),
                          "disagree_rate": round(float((g.role != g.role_audit).sum() / max(1, len(g[g.rule != "keep_cue"]))), 4),
                          "cue_public_self_dais": int(((g.role == "public") & (g.role_audit == "dais")).sum()),
                          "cue_dais_self_public": int(((g.role == "dais") & (g.role_audit == "public")).sum()),
                          "rationale_flagged": int(g.rationale_flag.sum())})

    # group on duplicated keys so the grouping columns stay visible inside agg()
    k = d.assign(_route=d.method, _body=d.body, _role=d.role)
    overall = pd.DataFrame([agg(d)]).assign(scope="all").set_index("scope")
    by_route = k.groupby("_route").apply(agg)
    by_body = k.groupby("_body").apply(agg).sort_values("n_turns", ascending=False)
    by_route_role = k.groupby(["_route", "_role"]).apply(agg)
    overall.to_csv(OUT / "disagreement_overall.csv")
    by_route.to_csv(OUT / "disagreement_by_route.csv")
    by_body.to_csv(OUT / "disagreement_by_body.csv")
    by_route_role.to_csv(OUT / "disagreement_by_route_role.csv")
    flagged = d[(d.rationale_flag == 1) | ((d.role == "public") & (d.role_audit == "dais"))]
    flagged[["episode_id", "cluster", "turn_idx", "body", "method", "family", "start_s", "end_s", "words",
             "role", "self_id", "role_audit", "rationale_flag", "stance_median", "rationale"]] \
        .sort_values(["body", "episode_id", "start_s"]).to_csv(OUT / "flagged_public_turns.csv", index=False)
    print(overall.to_string()); print(); print(by_route.to_string()); print()
    print(by_body.head(12).to_string()); print()
    print("cluster roll-up changes:",
          d.groupby(["role", "cluster_role_audit"]).size().to_dict())
    cl = d.drop_duplicates(["episode_id", "cluster"])
    print("clusters:", len(cl), "changed:", int((cl.role != cl.cluster_role_audit).sum()),
          "| rules:", cl.cluster_rule.value_counts().to_dict())
    print("rationale-flagged public turns:", int(d.rationale_flag.sum()),
          "| of which self_id unknown (rule keeps them public):",
          int(((d.rationale_flag == 1) & (d.rule == "keep_cue")).sum()))
    return flagged


def write_db(d):
    con = state.connect(None, create=False)
    con.execute(DDL)
    pvs = sorted(d.prompt_version.unique())
    con.executemany("DELETE FROM role_audit WHERE prompt_version = ? AND audit_version = ?",
                    [(p, AUDIT_VERSION) for p in pvs])
    rows = [(r.episode_id, r.cluster, int(r.turn_idx), r.prompt_version, r.role, r.self_id, r.role_audit,
             r.rule, int(r.rationale_flag), r.cluster_role_audit, r.cluster_rule, AUDIT_VERSION,
             pd.Timestamp.utcnow().isoformat())
            for r in d.itertuples()]
    con.executemany("INSERT OR REPLACE INTO role_audit VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)", rows)
    con.commit(); con.close()
    print(f"wrote {len(rows)} role_audit rows ({AUDIT_VERSION})")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="measure and write CSVs, do not touch the DB")
    ap.add_argument("--frame", default=None, help=f"frame CSV for route/roles_source (default {FRAME})")
    a = ap.parse_args(argv)
    d = apply_rule(load(a.frame))
    report(d)
    if not a.dry_run:
        write_db(d)
    return 0


if __name__ == "__main__":
    sys.exit(main())
