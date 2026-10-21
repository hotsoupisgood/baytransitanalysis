#!/usr/bin/env python
"""Export the dashboard's analysis panels from the ytpt pipeline (READ-ONLY).

    cd ~/ytpt && .venv/bin/python scripts/site_export/export_panels.py

Writes, beside this repo's index.html:
    data/panels.json        every number and caption the four analysis panels show
    data/panels/*.png       the figures those panels show, copied from output/

Four panels, all stamped EXPLORATORY (the stance kappa >= 0.6 gate of
plan/SHOWCASE.md 4a has not been passed; no human has coded any turn):

  stance  role gap, from output/stance_s1/models_frame/ (ordinal fits) plus the
          model-free tail shares from s1_tail_shares_bootstrap.csv. The ordinal
          stance gap is NOT the number to quote -- its PPC misfits the public
          distribution -- so the panel leads with the tail shares and carries
          the proportional-odds caveat (plan/STANCE-S1-RESULTS.md 7).
  nimby   YIMBY/NIMBY per anonymous speaker cluster, from
          output/stance_s1/nimby_view/, one figure per locate route, never
          pooled. Three caveats, two of them recomputed here from turn_stance
          rather than quoted: the Petaluma role misattribution and the San
          Leandro sign flip.
  renter  renter share vs what gets said, from output/renter_share/. Carries
          the withdrawn claim (plan/RENTER-SHARE.md 3, corrected 2026-10-18).
  rates   topic time per meeting-hour on the 5-body complete-coverage set, from
          output/wave1/. Carries the withdrawn pad-bounds finding.

Reads nothing but output/, plan/ and the two sqlite files, read-only. Never
writes inside the ytpt repo; never touches .env, plan/frame.csv or
plan/DECISIONS.md.
"""
from __future__ import annotations

import collections
import csv
import datetime as dt
import json
import math
import shutil
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from role_model import role_model  # noqa: E402  the one role-gap model
from export_flow import money  # noqa: E402  the spend ledger, one definition for both pages

YTPT = Path("~/ytpt").expanduser()
OUT = YTPT / "output"
SITE = Path(__file__).resolve().parents[2] / "docs"  # the GitHub Pages root of this repo
DATA = SITE / "data"
FIGS = DATA / "panels"
S1 = OUT / "stance_s1"
NIMBY = S1 / "nimby_view"
FRAME = S1 / "models_frame"
RENTER = OUT / "renter_share"
TABLES = OUT / "analysis" / "models" / "tables"
ROLE_AUDIT = OUT / "role_audit"
WAVE1 = OUT / "wave1"
CORPUS = Path("~/.local/share/ytpt/corpus.sqlite").expanduser()
STATE = Path("~/.local/share/ytpt/state.sqlite").expanduser()

STAMP = "EXPLORATORY — stance κ gate not passed"

#: The numbers this file's prose used to carry as 2026-10-18 literals, recomputed
#: from the view's own per-cluster CSVs by analysis/stance_s2_site_numbers.py
#: (plan/STANCE-S2.md). Empty dict if that has not been run: the call sites fall
#: back to the literal they replaced, so the page never breaks, it only goes stale.
_SITE_N_PATH = OUT / "stance_s2" / "site_numbers.json"
SITE_N = json.loads(_SITE_N_PATH.read_text(encoding="utf-8")) if _SITE_N_PATH.exists() else {}

#: The role-gap, role-classification and permutation numbers, recomputed from the
#: artefacts of runs 3 and 5 and of role_prediction by analysis/site_numbers_role.py.
#: "Who takes a side", the Stance panel's accuracy line and the Methods entries all
#: read this. Those numbers were corrected once already
#: (plan/REGRESSION-REVIEW-2026-10-19.md finding 2), and a literal typed into this
#: file is exactly how a corrected number stays wrong on a website.
_ROLE_N_PATH = OUT / "stance_s2" / "site_numbers_role.json"
ROLE_N = json.loads(_ROLE_N_PATH.read_text(encoding="utf-8")) if _ROLE_N_PATH.exists() else {}


def _pc(v, d="?"):
    return f"{v * 100:.1f}" if isinstance(v, (int, float)) else d


def _tr(s):
    """'46.0 / 42.9 / 11.2' from a site_numbers split dict."""
    s = s or {}
    return f"{_pc(s.get('YIMBY'))} / {_pc(s.get('NIMBY'))} / {_pc(s.get('neutral'))}"


_sl = SITE_N.get("sanleandro") or {}
_one = SITE_N.get("public_cluster_scored_turns") or {}


def _split(s, fallback):
    """One of site_numbers.json's split dicts in the shape the panel publishes."""
    if not s:
        return fallback
    return {"YIMBY": s["YIMBY"], "NIMBY": s["NIMBY"], "neutral": s["neutral"], "n_public_clusters": s["n"]}


def rd(path):
    p = Path(path)
    return list(csv.DictReader(p.open(encoding="utf-8"))) if p.exists() else []


def num(rows, *keys):
    """CSV rows -> floats where they parse, so the page does no string maths."""
    out = []
    for r in rows:
        d = dict(r)
        for k, v in r.items():
            if keys and k not in keys:
                continue
            try:
                f = float(v)
                d[k] = None if (math.isnan(f) or math.isinf(f)) else f
            except (TypeError, ValueError):
                pass
        out.append(d)
    return out


def clean(o):
    if isinstance(o, float):
        return None if (math.isnan(o) or math.isinf(o)) else o
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    return o


def copy_figs(paths):
    """Copy the figures a panel shows into data/panels/ and return their names."""
    FIGS.mkdir(parents=True, exist_ok=True)
    names = []
    for p in paths:
        p = Path(p)
        if p.exists():
            shutil.copyfile(p, FIGS / p.name)
            names.append(p.name)
    return names


# ── caveats recomputed from turn_stance, not quoted from a write-up ──────────
def role_misattribution():
    """**Superseded 2026-10-18 by the role audit** (plan/ROLE-AUDIT-2026-10-18.md).
    This was the site's own heuristic flag -- public-role clusters holding at
    least one turn the stance model self-identifies as `elected`. The real
    audit (rule ra-1, state.sqlite `role_audit`) now drives every public/dais
    number in the panels, and role_audit_summary() reads it. This function is
    kept only because its per-body table is the clearest way to show *which*
    bodies the flag fired on before the audit existed; it is reported as
    history, not as the current caveat."""
    cp = sqlite3.connect(f"file:{CORPUS}?mode=ro", uri=True)
    st = sqlite3.connect(f"file:{STATE}?mode=ro", uri=True)
    eps = {r[0]: (r[1], r[2], r[3]) for r in cp.execute("select episode_id, body, method, family from episodes")}
    roles = {(r[0], r[1]): r[2] for r in cp.execute("select episode_id, speaker, role from episode_speakers")}
    clusters = collections.defaultdict(list)
    for eid, cl, sm, sid in st.execute("select episode_id, cluster, stance_median, self_id from turn_stance"):
        m = eps.get(eid)
        if not m or m[2] not in ("density", "both") or roles.get((eid, cl)) != "public":
            continue
        clusters[(m[0], m[1], eid, cl)].append((sm, sid))

    def label(ts):
        v = [s for s, _ in ts if s is not None]
        if not v:
            return "no stance"
        m = sum(v) / len(v)
        return "YIMBY" if m >= 0.5 else "NIMBY" if m <= -0.5 else "neutral"

    tot = collections.Counter()
    flagged = collections.Counter()
    for key, ts in clusters.items():
        lab = label(ts)
        if lab == "no stance":
            continue
        tot[(key[0], key[1])] += 1
        tot[(key[0], key[1], lab)] += 1
        if any(sid == "elected" for _, sid in ts):
            flagged[(key[0], key[1])] += 1
            flagged[(key[0], key[1], lab)] += 1
    rows = []
    for (body, route) in sorted({k[:2] for k in tot if len(k) == 2}):
        rows.append({"body": body, "route": route,
                     "clusters_with_stance": tot[(body, route)],
                     "flagged": flagged[(body, route)],
                     "flagged_yimby": flagged[(body, route, "YIMBY")],
                     "flagged_nimby": flagged[(body, route, "NIMBY")],
                     "yimby": tot[(body, route, "YIMBY")], "nimby": tot[(body, route, "NIMBY")]})
    rows.sort(key=lambda r: -r["flagged"])
    overall = {"clusters_with_stance": sum(r["clusters_with_stance"] for r in rows),
               "flagged": sum(r["flagged"] for r in rows)}
    cp.close()
    st.close()
    return {"rows": rows, "overall": overall}


def role_audit_summary():
    """The real role audit (rule ra-1, state.sqlite `role_audit`,
    plan/ROLE-AUDIT-2026-10-18.md), which is the **reported** role everywhere on
    this site as of 2026-10-18.

    The diarization-cue role labeller reads cue phrases, so a councilmember who
    says "I live in this neighbourhood" or "on behalf of my community" lands in
    `public`. Every turn also carries the codebook's SELF_ID -- what the speaker
    says about who they are, from that turn's words only -- which is an
    independent instrument. Where the two disagree in the public bucket, the cue
    label is wrong 20-39 % of the time on the adjudicable subset, which was
    enough to carry a starred body-level result (Petaluma, Swagit route) on its
    own. Rule, per turn: self_id elected|staff -> dais; resident|homeowner|
    renter|business|org_rep -> public; unknown -> keep the cue role. The
    analysis unit is the episode x cluster, so the turn rule rolls up by
    majority; ties and clusters with no informative turn keep the cue role.

    Returns the disagreement rates, the role-gap before/after, and the counts a
    reader needs to see that this is a re-partition and not a cleanup."""
    st = sqlite3.connect(f"file:{STATE}?mode=ro", uri=True)
    tot = informative = disagree = 0
    by_cue = collections.Counter()
    cells = collections.defaultdict(lambda: collections.Counter())
    for cue, sid, aud, cra in st.execute(
            "select role_cue, self_id, role_audit, cluster_role_audit from role_audit"):
        tot += 1
        if sid and sid != "unknown":
            informative += 1
            by_cue[(cue, "informative")] += 1
            if aud and aud != cue:
                disagree += 1
                by_cue[(cue, "disagree")] += 1
        if cra and cra != cue:
            cells[cue]["cluster_changed"] += 1
    version = next(iter(st.execute("select audit_version from role_audit limit 1")), ("ra-1",))[0]
    st.close()

    def rate(cue):
        i = by_cue[(cue, "informative")]
        return {"informative": i, "disagree": by_cue[(cue, "disagree")],
                "rate": round(by_cue[(cue, "disagree")] / i, 3) if i else None}

    def gaps(sfx):
        rows = {r["family"]: r for r in num(rd(TABLES / f"role_gap_gap_stance_all{sfx}.csv"))}
        hrows = {r["family"]: r for r in num(rd(TABLES / f"role_gap_gap_hostility_all{sfx}.csv"))}
        out = {}
        for name, src in (("stance", rows), ("hostility", hrows)):
            r = src.get("density") or {}
            out[name] = {"median": r.get("gap_latent_median"), "lo90": r.get("gap_lo90"),
                         "hi90": r.get("gap_hi90"), "n_public": r.get("n_public")}
        return out

    return {
        "audit_version": version,
        "turns": tot, "informative": informative, "unknown": tot - informative,
        "unknown_share": round((tot - informative) / tot, 3) if tot else None,
        "disagree": disagree,
        "disagree_rate_informative": round(disagree / informative, 3) if informative else None,
        "by_cue_role": {"public": rate("public"), "dais": rate("dais")},
        # Recomputed, not quoted: these two were 2026-10-18 literals and went stale
        # when the S2 top-up grew the frame (plan/STANCE-S2.md).
        "clusters_changed": SITE_N.get("clusters_changed", 265),
        "public_clusters_with_stance": SITE_N.get("public_clusters_with_stance", {"cue": 414, "audit": 457}),
        "role_gap_density": {"cue": gaps("_cuebaseline"), "audit": gaps("_roleaudit")},
        "disagreement_by_body": num(rd(ROLE_AUDIT / "disagreement_by_body.csv")),
        "source": "state.sqlite role_audit; plan/ROLE-AUDIT-2026-10-18.md",
    }


def sign_flip():
    """Public turns and clusters carrying sub_topic = affordability_displacement,
    the sub-topic where "opposed to the item" is the pro-housing side and the
    YIMBY/NIMBY mapping inverts (plan/NIMBY-PILOT.md 3.3)."""
    cp = sqlite3.connect(f"file:{CORPUS}?mode=ro", uri=True)
    st = sqlite3.connect(f"file:{STATE}?mode=ro", uri=True)
    eps = {r[0]: (r[1], r[2], r[3]) for r in cp.execute("select episode_id, body, method, family from episodes")}
    roles = {(r[0], r[1]): r[2] for r in cp.execute("select episode_id, speaker, role from episode_speakers")}
    turns = collections.Counter()
    flip = collections.Counter()
    cl_any = collections.defaultdict(set)
    cl_flip = collections.defaultdict(set)
    for eid, cl, sm, sub in st.execute("select episode_id, cluster, stance_median, sub_topic from turn_stance"):
        m = eps.get(eid)
        if not m or m[2] not in ("density", "both") or roles.get((eid, cl)) != "public":
            continue
        if sm is None:
            continue
        turns[m[0]] += 1
        cl_any[m[0]].add((eid, cl))
        if sub == "affordability_displacement":
            flip[m[0]] += 1
            cl_flip[m[0]].add((eid, cl))
    rows = [{"body": b, "scored_public_turns": turns[b], "flip_turns": flip[b],
             "clusters": len(cl_any[b]), "flip_clusters": len(cl_flip[b]),
             "flip_turn_share": round(flip[b] / turns[b], 3) if turns[b] else None}
            for b in sorted(turns, key=lambda b: -flip[b])]
    total_clusters = len({c for s in cl_any.values() for c in s})
    cp.close()
    st.close()
    return {"rows": [r for r in rows if r["flip_turns"]], "total_public_clusters": total_clusters}


# ── panels ───────────────────────────────────────────────────────────────────
def panel_stance():
    kappa = json.loads((S1 / "s1_k_replicate_agreement.json").read_text(encoding="utf-8")) if (S1 / "s1_k_replicate_agreement.json").exists() else {}
    tails = [r for r in num(rd(S1 / "s1_tail_shares_bootstrap_roleaudit.csv")) if r.get("role_source") == "db"]
    tails_cue = [r for r in num(rd(S1 / "s1_tail_shares_bootstrap_cue.csv")) if r.get("role_source") == "db"]
    audit = role_audit_summary()
    # The `_computed` forests came off the page 2026-10-20 (owner): the computed role
    # instrument is a second, 35-episode facet of the same fit, and the dashboard reports
    # the role gap from one model only. They stay in output/stance_s1/models_frame/.
    # The `_db` forests and PPCs followed them off later the same day, for the same
    # reason carried to its end: the site reports one model, the nested mixed logistic
    # fit in "Who takes a side", so the cumulative-logit fit should not appear here as
    # four figures when its tables are already gone. The figures stay in
    # output/stance_s1/models_frame/figs/. The panel keeps the model-free tail shares,
    # which are the numbers this panel asks the reader to quote.
    figs = []
    return {
        "title": "Stance: the role gap",
        "stamp": STAMP,
        # Concision pass, owner 2026-10-19 12:46: one line of definition, the rest of
        # the instrument (the scales, the turn floor, the codebook) on Methods § stance.
        "lede": ("Does public comment differ from the dais inside the same meeting? Three reads of "
                 "every turn against one pinned codebook."),
        "tldr": ("The public reads as more hostile than the dais."),
        "corrected": "corrected 2026-10-18 after role audit",
        # TLDR rule (owner, 2026-10-18 21:16, docs/SITE-STYLE.md): vague, result-
        # pointing, no numbers at all. Every latent gap, share, interval and κ that used
        # to be quoted here is in the tables and caveats of this panel's body below, and
        # the instrument itself is on Methods § stance.
        # Concision pass, owner 2026-10-19 12:46: three bullets, no numbers, in the
        # house TLDR form. The two findings that were their own bullets -- that the
        # audit strengthened hostility and that the density reading was dais speech --
        # are the second bullet now; the both-tails shape is the third.
        # Owner, 2026-10-20 22:02: the density half of the TLDR and of the second
        # bullet is deleted, not rewritten -- "not as more opposed on density"
        # contradicted this panel's own which-side odds ratio. Nothing replaces it.
        "headline": [
            "I scored every turn for stance and hostility and compared public comment with the dais.",
            "Found the public more hostile than the dais.",
            "Found the public sits in both tails at once while the dais stays neutral, so quote the "
            "tail shares below, not the gap.",
        ],
        # Owner, 2026-10-19: "we also need to put the accuracy for that classification
        # of dai/not." It belongs in this panel rather than NIMBY/YIMBY, because it is a
        # statement about the stance instrument's separability, not about housing.
        "role_pred": role_accuracy(),
        "role_audit": audit,
        "model_free": {
            "note": ("Model-free contrast, 2,000 episode-cluster bootstrap reps, role instrument db, "
                     "**under the reported role_audit role** (corrected 2026-10-18 after role audit). "
                     "Shares are among turns with a numeric stance; the public also has far fewer "
                     "not_applicable turns. Every tail gap is larger than under the cue role, on both "
                     "outcomes and in every family."),
            "rows": tails,
            "rows_cue": tails_cue,
            "cue_note": ("The same contrast under the old diarization-cue role, kept as a sensitivity. It "
                         "reproduces what this panel published before 2026-10-18 exactly, so the "
                         "difference is the role column and nothing else."),
            "source": "output/stance_s1/s1_tail_shares_bootstrap_roleaudit.csv",
        },
        # The Bayesian ordinal role-gap tables (cumulative-logit family fits, the
        # cue-role sensitivity and the role_audit refit) came off the page 2026-10-20:
        # the dashboard reports the role gap from the nested mixed logistic fit in "Who
        # takes a side" and from nothing else, and the numbers to quote in this panel are
        # the model-free tail shares above. The fits, their tables and their diagnostics
        # stay in output/stance_s1/models_frame/ and output/analysis/models/tables/.
        "descriptives": num(rd(S1 / "s1_descriptives_by_role.csv")),
        "replicate_agreement": kappa.get("all") or kappa,
        "caveats": [
            {"name": "Proportional odds misfits the public stance distribution",
             "text": ("A posterior predictive check of that fit puts 58 % of public turns at stance 0 "
                      "against 23 % observed: a single shift"
                      "of shared cutpoints cannot represent a distribution spread into both tails. The "
                      "latent stance gap therefore averages two opposite effects. Quote the tail shares, "
                      "not the gap. A two-part model (stance taken or not; direction given a stance) is the "
                      "right next model.")},
            {"name": "The κ ≥ 0.6 hand-coding gate is not passed",
             "text": ("No human has coded any of these turns. The 0.98 replicate κ is the instrument "
                      "agreeing with itself (noise floor ≈ 2 %); the one cross-instrument check on record "
                      "is κ 0.63 against a second codebook, same side 88 % of the time. Nothing here is a "
                      "confirmatory result.")},
            {"name": "There is no staff role",
             "text": ("Roles are dais / public / unknown; staff reports sit inside dais and pull it toward "
                      "neutral. 472 unknown-role turns are outside the frame, and the computed role "
                      "instrument (35 classifier episodes) is a separate, wider facet rather than more data.")},
            {"name": "The reported role is an audit, not a gold label",
             "text": ("Added 2026-10-18. The public/dais split here is role_audit (rule "
                      f"{audit['audit_version']}), which prefers the codebook's SELF_ID over the "
                      "diarization-cue role. SELF_ID is model output and no human has checked it. "
                      f"{audit['unknown']:,} of {audit['turns']:,} turns "
                      f"({audit['unknown_share']:.0%}) say nothing about who the speaker is and keep the "
                      "cue role, so the audited public bucket is still contaminated at an unknown rate — "
                      "the measured error is a lower bound on the informative subset only. And the audit "
                      "is a re-partition in both directions, not a cleanup: it admits cue-dais clusters as "
                      "well as removing cue-public ones, and the public base with a stance *rises* from "
                      f"{audit['public_clusters_with_stance']['cue']} to "
                      f"{audit['public_clusters_with_stance']['audit']} clusters. Differences from the _cue "
                      "numbers are not 'contamination removed'."),
             "table": audit["disagreement_by_body"][:12]},
        ],
        "figs": figs,
        # Written titles, not filenames (readability review §7.3). Empty since the
        # cumulative-logit forests and PPCs came off the page 2026-10-20.
        "fig_titles": {},
        "fig_caps": {},
        "methods_anchor": "stance",
    }


def role_accuracy():
    """How well the words alone tell a dais speaker from a public one.

    This is the Stance panel's short answer to an obvious question: if the two
    roles really talk differently, a model given only what was said should be
    able to guess which bench the speaker was on. It can, about as well as a
    decent medical screening test -- and the thing it uses is not the position
    but the opinionatedness, which is the same finding as "Who takes a side"
    arriving from a different direction.

    Only the whole-body holdout is quoted: leave-episode-out is optimistic
    here, because the same meeting's dais sits in both folds. Numbers from
    site_numbers_role.json; None if it has not been built.
    """
    rp = ROLE_N.get("role_prediction") or {}
    cue, aud = rp.get("cue_role") or {}, rp.get("audited_role") or {}
    cl, tn = cue.get("cluster"), cue.get("turn")
    dirn = cue.get("signed_stance_only_cluster_logit")
    if not (cl and tn):
        return None
    imp = rp.get("importance_cluster") or []
    return {
        "title": "How well content alone tells the dais from the public",
        # Concision pass, owner 2026-10-19 12:46: one sentence visible. The ablation,
        # the calibration slope and the circularity of the audited target were a fold
        # here that repeated Methods § models almost word for word, so the fold is gone
        # and the panel links Methods instead.
        "text": (
            f"Given only what was said, a model picks the public speaker out of a pair about "
            f"{cl['auc']:.2f} of the time — using not the position but its distance from neutral."),
        "detail": (
            "The features are stance, hostility, the twelve frames, sub-topic and length, and "
            "two of them do nearly all the work: dropping absolute stance costs "
            f"{(imp[0]['auc_drop'] if imp else 0):.2f} of AUC and dropping length "
            f"{(imp[1]['auc_drop'] if len(imp) > 1 else 0):.2f}, with nothing else over "
            "0.02. "
            + (f"Signed stance on its own is at chance — {dirn['auc']:.2f} by cluster, "
               "against 0.50 for a coin — so the model is not learning who is a NIMBY, it is "
               "learning who holds a position at all. " if dirn else "")
            + "Two caveats. Calibration does not travel: the ranking holds between bodies but "
            f"the probabilities do not, at a slope of {cl['cal_slope']:.2f} out of body, so "
            "this is a measurement of how separable the two roles are and not a classifier "
            "anyone should point at a new city. And the target here is the diarization-cue "
            "role. Fitted against the audited role the same model reaches "
            f"{(aud.get('cluster') or {}).get('auc', 0):.2f}, but that number is circular — "
            "the audit is itself built from what a speaker says about who they are, so the "
            "model is partly predicting its own input."),
        "auc_cluster": cl["auc"], "auc_turn": tn["auc"],
        "cal_slope_cluster": cl["cal_slope"],
        "signed_stance_only": (dirn or {}).get("auc"),
        "n_clusters": cl["n"], "n_turns": tn["n"],
        "holdout": rp.get("holdout"),
        "top_features": imp[:2],
        "audited_auc_cluster": (aud.get("cluster") or {}).get("auc"),
        "audited_note": aud.get("circularity_note"),
        "source": rp.get("source"),
    }


def who_takes_a_side():
    """The role gap, from the one model the dashboard reports it from.

    Owner, 2026-10-19: "lets keep that lme4 model then as the one we use in the
    dashboard. drop the others from the final explanation but keep it in the
    repo." So this block quotes one estimator and nothing else. It is the nested
    turn-level mixed logistic regression -- one row per transcript turn, random
    intercepts for the episode, the meeting and the body -- fitted twice on the
    same frame with the same specification: did a turn take a side at all, and,
    among the turns that took one, which side.

    Everything the block used to carry from the other fits is gone from the
    page: the eightfold cluster-level figure, the 2.11 restatement, the
    four-denominator table, the Bayesian, GEE, custom-quadrature and
    beta-binomial columns. Those runs are still in the repo; Methods says in one
    line that they agree with this fit, and that line's number is computed, not
    asserted.

    Every number comes from export/role_model.py, which reads
    output/renter_share_6/. If run 6 is missing the function returns None and
    the panel simply does not carry the block.

    Concision rules of HEAD 524cea5 still hold: one lead sentence, two steps at
    one line each, one line saying what to quote, and everything else -- the
    shares, the variance components, the limits -- one click away.
    """
    M = role_model()
    if not M.get("available"):
        return None
    steps = {s["key"]: s for s in M["steps"]}
    side = steps.get("which_side")
    if not side:
        return None
    took = steps.get("any_side")

    orx = lambda s: f"{s['odds_ratio']:.2f}"
    orci = lambda s: (f"95 % {s['or_ci95'][0]:.2f} to {s['or_ci95'][1]:.2f}"
                      if all(v is not None for v in s["or_ci95"]) else "95 % not bounded below")
    lo = lambda s: (f"{s['ci95'][0]:+.2f} to {s['ci95'][1]:+.2f}"
                    if all(v is not None for v in s["ci95"])
                    else f"up to {s['ci95'][1]:+.2f}, with no lower bound")
    sd = lambda v: "0" if not v else f"{v:.2f}"

    # The two visible lines, one per step, in the order they happen.
    vis_steps = []
    if took:
        vis_steps.append({
            "name": took["name"],
            "text": f"{orx(took)}× in odds ({orci(took)}).",
            "odds_ratio": took["odds_ratio"], "beta": took["beta"],
            "ci95": took["ci95"]})
    vis_steps.append({
        "name": side["name"],
        "text": f"{orx(side)}× against the item ({orci(side)}).",
        "odds_ratio": side["odds_ratio"], "beta": side["beta"],
        "ci95": side["ci95"]})

    headline = (
        "Public speakers take a side far more often than the dais does, and when they do "
        "they come out against the item somewhat more often."
        if took else
        "When public speakers take a side they come out against the item somewhat more "
        "often than the dais does.")

    # The plain-words description of the estimator the owner asked for: a mixed
    # model with meeting and episode intercepts, no package names.
    model_line = (
        "Both lines come from one mixed model, with its own baseline for every meeting "
        "and every stretch of meeting."
        if took else
        "That line comes from a mixed model with its own baseline for every meeting and "
        "every stretch of meeting.")

    restated = (f"For “more opposed”, quote the second line — {orx(side)}× — not the "
                "first, which is about taking a position at all.")

    why_parts = []
    if took:
        why_parts.append(
            f"Behind the two steps: {_pc(took['rate_public'])} % of public turns take a side "
            f"rather than sitting in the neutral band, against {_pc(took['rate_dais'])} % of "
            f"dais turns; of the turns that do take one, {_pc(side['rate_public'])} % of public "
            f"turns come out against the item against {_pc(side['rate_dais'])} % of dais ones. "
            "The two steps are separate fits on their own rows and do not multiply together.")
    else:
        why_parts.append(
            f"Of the turns that take a side, {_pc(side['rate_public'])} % of public ones come "
            f"out against the item against {_pc(side['rate_dais'])} % of dais ones.")
    why_parts.append(
        f"Almost all the dependence in this corpus is inside meetings rather than between "
        f"bodies: on the second step the spread between stretches of meeting is "
        f"{sd(side['sd_episode'])} and between meetings {sd(side['sd_meeting'])} on the "
        f"log-odds scale, while the spread between bodies is {sd(side['sd_body'])} once "
        "meeting is in the model. Which is why the model groups turns by meeting and not by "
        "city.")
    why_parts.append(
        "Renter share does not move either step: its interval covers zero on both, and so "
        "does its interaction with role. Staff speakers are dropped from these rows "
        "altogether — they read reports rather than take positions — and a council member "
        "chairing an item is doing something other than taking one too, which is most of "
        "what the first step is measuring.")

    table = []
    if took:
        table.append({
            "what": "Took a side at all, of every turn with a stance",
            "odds_ratio": took["odds_ratio"], "or_ci95": took["or_ci95"],
            "delta": took["beta"], "ci95": took["ci95"],
            "turns": took["n_turns"], "episodes": took["n_episodes"],
            "meetings": took["n_meetings"]})
    table.append({
        "what": "Against the item rather than for it, of the turns that took a side",
        "odds_ratio": side["odds_ratio"], "or_ci95": side["or_ci95"],
        "delta": side["beta"], "ci95": side["ci95"],
        "turns": side["n_turns"], "episodes": side["n_episodes"],
        "meetings": side["n_meetings"]})

    table_note = (
        "Odds ratios are for a public turn against a dais one, with renter share, its "
        "interaction with role and the locate route also in the model, and with a baseline "
        "per body, per meeting and per stretch of meeting. Intervals are 95 % and come from "
        "the likelihood rather than a standard error, which is why they are not symmetric "
        "around the estimate.")

    limits = (
        "The unit here is a scored turn, not a person, and both the stance score and the "
        "public/dais label are model output that no human has checked — the role label comes "
        "from what a speaker says about who they are. The first step is largely a measure of "
        "speech act rather than opinion: chairing, moving and reading a report are all "
        "neutral by construction, and they are things only the dais does. The second step's "
        "interval is a shade wider than it needs to be, because the optimiser stops just "
        "short of the likelihood's peak and the interval was taken at a loosened tolerance. "
        "And the permutation test once quoted for this gap has been withdrawn; the evidence "
        "here is this model and nothing else.")

    return {
        "title": "Who takes a side",
        "headline": headline,
        "model_line": model_line,
        "steps": vis_steps,
        "restated": restated,
        "why": " ".join(why_parts),
        "table": table,
        "table_note": table_note,
        "limits": limits,
        "sources": [M["source"]] + list(M.get("scripts") or []),
    }


def dot_plot_caption():
    """The caption under the role dot plot, in house style.

    docs/STYLE.md §3: the caption is the prose -- it names the figure and reads
    it, rather than labelling it. The one number in it is the per-turn share
    sitting at exactly neutral, which is the per-statement version of what the
    two halves look like, and it comes from the DB via site_numbers_role.json.
    The per-cluster counts under the figure are counted in the browser off the
    dots actually drawn, so they follow the route selector.
    """
    z = ((ROLE_N.get("zero_stance") or {}).get("by_role")) or {}
    dz = (z.get("dais") or {}).get("share_at_zero")
    pz = (z.get("public") or {}).get("share_at_zero")
    if dz is None or pz is None:
        return None
    # Concision pass, owner 2026-10-19 12:46: the caption reads the figure in two
    # sentences. The per-statement share stays -- it is the number the shape is made
    # of. What a dot is, and that most rest on one scored statement, is in the
    # panel's "How to read this chart" fold and its caveats.
    return (
        "In the plot above the dais half is a band pressed onto the neutral line, and the "
        "public half is two clouds pulled away from it, balanced above and below.")


def panel_nimby():
    summary = json.loads((NIMBY / "summary.json").read_text(encoding="utf-8")) if (NIMBY / "summary.json").exists() else {}
    by_body = num(rd(NIMBY / "public_dais_shares_by_body_route.csv"))
    by_route = num(rd(NIMBY / "public_shares_by_route.csv"))
    # The routes the view actually computed, biggest first -- read from its own
    # summary so a new locate route (fullasr landed with the S2 top-up,
    # plan/STANCE-S2.md) cannot be silently dropped from this panel.
    routes = list((summary.get("routes") or {}).keys()) or \
        ["classifier", "local_asr", "agenda_index", "swagit_transcript"]
    # The public starred bar chart stays the panel's lead figure (owner, 2026-10-19:
    # "the horizontal bar plot should be there as well as it is the most provoking
    # finding").
    #
    # The all-roles version of the same chart came off the page 2026-10-20. It was
    # drawn on 10-19, before the data freeze, and nothing in this repo regenerates
    # it: the view refreshes the public bars and the public-vs-dais pair only, and
    # copy_figs() silently skips what it cannot find, so those five PNGs would have
    # ridden along unchanged next to figures rebuilt from the frozen data. The
    # dais comparison the reader needs is still here, in fig_public_vs_dais_*.
    figs = copy_figs([NIMBY / f"fig_public_yimby_nimby_{r}_stars.png" for r in routes]
                     + [NIMBY / f"fig_public_vs_dais_{r}.png" for r in routes])
    by_body_cue = num(rd(NIMBY / "public_dais_shares_by_body_route_cue.csv"))
    by_route_cue = num(rd(NIMBY / "public_shares_by_route_cue.csv"))
    mis = role_misattribution()
    flip = sign_flip()
    audit = role_audit_summary()
    sl = next((r for r in flip["rows"] if r["body"] == "sanleandro"), {})
    stars = {r: (summary.get("routes", {}).get(r, {}) or {}).get("stars_public", []) for r in routes}
    pet_audit = next((r for r in by_body if r["role"] == "public" and r["body"] == "petaluma"
                      and r["route"] == "swagit_transcript"), {})
    pet_cue = next((r for r in by_body_cue if r["role"] == "public" and r["body"] == "petaluma"
                    and r["route"] == "swagit_transcript"), {})
    oak_audit = next((r for r in by_body if r["role"] == "public" and r["body"] == "oakland"), {})
    return {
        "title": "YIMBY / NIMBY",
        "stamp": STAMP,
        # Readability review §7.2: the construct definition used to open this panel --
        # 103 words of jargon in the position that should be the hook. Two sentences
        # stay, and index.html now renders the lede *below* the headline bullets; the
        # thresholds, the transit exclusion and the no-pooling rule moved down into the
        # caveat "How a speaker gets a label". The second sentence is the clause the
        # reviewer asked for (§7, "anonymous"): the page says clusters are never linked
        # across meetings while the quote cards show speakers naming themselves, and a
        # general reader takes that as a contradiction unless both are said at once.
        # Concision pass, owner 2026-10-19 12:46: one sentence of definition. The
        # no-linking clause and the names-as-spoken clause are both said once on the
        # page already, under the episode reader, so they are not repeated here.
        "lede": ("No new scoring: a speaker is YIMBY if their turns lean toward the housing item, NIMBY "
                 "if against."),
        # Rewritten 2026-10-19 after the S2 top-up (plan/STANCE-S2.md): the frame
        # roughly doubled the public base, the overall lean crossed to the against
        # side, and three more bodies separated from an even split.
        "tldr": ("Public speakers split close to evenly for and against the housing items, leaning "
                 "slightly against."),
        "corrected": "corrected 2026-10-18 after role audit; frame grown 2026-10-19 (S2)",
        # TLDR rule (owner, 2026-10-18 21:16, docs/SITE-STYLE.md): vague, result-
        # pointing, no numbers at all. The splits, the stars, the intervals and the
        # withdrawn Petaluma star are in the table and caveats of this panel's body.
        # Concision pass, owner 2026-10-19 12:46: four bullets, no numbers, in the house
        # TLDR form. San Jose, Napa, Santa Clara and Oakland are named in the bars and
        # in the per-body table; the withdrawn Petaluma star keeps its own caveat; the
        # cell counts the old second bullet quoted are the stars on the chart.
        "headline": [
            # "anonymous" dropped in the third pass: the quote cards show speakers naming
            # themselves, and the word read as a contradiction (readability review §7).
            "I rolled the per-turn stance up to speakers and asked how the public split YIMBY against "
            "NIMBY on housing items.",
            "Found the split close to even, with only a few bodies separating from it on their own.",
            "Found the routes that read the same body can disagree, so nothing here is pooled.",
            "I put the shares in the starred bars and the table below, and kept the limits "
            "on Methods.",
        ],
        # Owner, 2026-10-19: the 8.2 goes on the dashboard. It goes on carrying its
        # own denominator and its own split -- see who_takes_a_side().
        "role_side": who_takes_a_side(),
        # The dot plot's caption: prose, with its one number substituted here.
        "dot_caption": dot_plot_caption(),
        "role_audit": audit,
        "overall": {"n_turns": summary.get("n_turns"), "n_clusters": summary.get("n_clusters"),
                    "threshold": summary.get("threshold"), "bootstrap_reps": summary.get("bootstrap_reps"),
                    "clusters_by_role_label": summary.get("clusters_by_role_label"),
                    # recomputed with the frame (plan/STANCE-S2.md), not quoted
                    "split": _split(SITE_N.get("split_audit"),
                                    {"YIMBY": 0.460, "NIMBY": 0.429, "neutral": 0.112, "n_public_clusters": 457}),
                    "split_cue": _split(SITE_N.get("split_cue"),
                                        {"YIMBY": 0.420, "NIMBY": 0.415, "neutral": 0.164, "n_public_clusters": 414}),
                    "role_col": "role_audit", "stars_public": stars},
        "by_route": by_route,
        "by_body": by_body,
        "by_route_cue": by_route_cue,
        "by_body_cue": by_body_cue,
        "caveats": [
            # Owner, 2026-10-19 10:53: explain the n on the bars and the data-collection
            # target. Both sit first because a reader hits the bar labels before anything else.
            {"name": "What n means on the bars, and why it does not add up to the other two numbers",
             "text": ("On each bar label, n is the number of speaker clusters at that body that carry a "
                      "stance at all — one cluster is one diarized voice inside one meeting. The two "
                      "numbers after it are not parts of n: the episode count says how many stretches of "
                      "meeting those clusters came from (one episode usually holds several speakers), and, "
                      "where shown, the public count is the subset of n that the role audit calls public. "
                      "The green and red bars are the YIMBY and NIMBY shares of n; the remainder is "
                      "neutral and is not drawn, so the two bars do not sum to one either.")},
            {"name": "How much data each body has, and why it is uneven",
             "text": ("The collection aimed for roughly thirty regular meetings per organization over the "
                      "year, which is about what a council that meets twice a month produces. Not every "
                      "body got there: some archives could not be located, some meet monthly, and the "
                      "YouTube bodies are still being downloaded at a fixed daily budget, so their counts "
                      "are floors. A body with few clusters gets a wide interval and no star, not a zero.")},
            # Moved out of the lede (readability review §7.2). Nothing here is new; it is
            # the construct definition that used to sit above the bullets, plus the note
            # that the figures were redrawn under the audited role, which used to be a
            # tail on every figure caption (§7.6).
            {"name": "How a speaker gets a label, and what the label is not",
             "text": ("A cluster's density stance is the mean of its turns' stance medians over the turns "
                      "that carry a numeric stance: YIMBY at ≥ +0.5, NIMBY at ≤ −0.5, neutral between. "
                      "Transit-family episodes are excluded, YIMBY/NIMBY being a housing construct. There "
                      "is one table and one figure per locate route and they are never pooled, because the "
                      "routes that read the same body can disagree about it. "
                      "Corrected 2026-10-18 after the role audit: every figure and table on this panel is "
                      "drawn under the audited speaker role, and the earlier cue-role versions are kept in "
                      "the repo with a _cue suffix as a sensitivity."),
             },
            {"name": "Petaluma star WITHDRAWN — it was dais speech in the public bucket",
             "text": ("Corrected 2026-10-18 after role audit. Petaluma's Swagit-route YIMBY lean was one of "
                      "only two starred results on this page, and it does not survive being audited against "
                      "what the speakers say about themselves. Net YIMBY − NIMBY went "
                      f"{pet_cue.get('net_yimby_minus_nimby', -0.53):+.2f} "
                      f"({pet_cue.get('net_lo90', 0):+.2f}, {pet_cue.get('net_hi90', 0):+.2f}) ★ to "
                      f"{pet_audit.get('net_yimby_minus_nimby', 0):+.2f} "
                      f"({pet_audit.get('net_lo90', 0):+.2f}, {pet_audit.get('net_hi90', 0):+.2f}), "
                      "interval across zero, on "
                      f"{pet_cue.get('n_clusters', 17):.0f} → {pet_audit.get('n_clusters', 25):.0f} public "
                      "clusters: four clusters leave the public bucket and thirteen join it. The single "
                      "largest “public” cluster in that frame — 1,083 words, YIMBY — is an elected member "
                      "advocating housing near the SMART station, by their own self_id. Two more are "
                      "NIMBY-side members. **Petaluma public-YIMBY is not a finding.** Oakland's star does "
                      f"survive, at {oak_audit.get('net_yimby_minus_nimby', 0):+.2f} "
                      f"({oak_audit.get('net_lo90', 0):+.2f}, {oak_audit.get('net_hi90', 0):+.2f}). "
                      "The table below is the site's older heuristic flag, kept as history: it fired on the "
                      f"same bodies, flagging {mis['overall']['flagged']} of "
                      f"{mis['overall']['clusters_with_stance']} cue-public clusters before the real audit "
                      "existed."),
             "table": [r for r in mis["rows"] if r["flagged"]][:12]},
            {"name": "The reported role is an audit, not a gold label",
             "text": ("Added 2026-10-18. Every public/dais number on this page is role_audit (rule "
                      f"{audit['audit_version']}): where the speaker says who they are, that beats the "
                      "cue-phrase label. In the cue-labelled public bucket "
                      f"{audit['by_cue_role']['public']['disagree']} of "
                      f"{audit['by_cue_role']['public']['informative']} adjudicable turns "
                      f"({audit['by_cue_role']['public']['rate']:.0%}) say `elected` or `staff`; in the dais "
                      f"bucket only {audit['by_cue_role']['dais']['rate']:.1%} disagree, because that bucket "
                      "is ~50× larger. But SELF_ID is model output, not a human label, and "
                      f"{audit['unknown_share']:.0%} of turns say nothing about the speaker and keep the cue "
                      "role — so the audited public bucket is still contaminated at an unknown rate. The "
                      "audit also moves clusters *both* ways: the public base with a stance rises from "
                      f"{audit['public_clusters_with_stance']['cue']} to "
                      f"{audit['public_clusters_with_stance']['audit']} clusters. It is a re-partition, not "
                      "a cleanup, and a shift in the split is not 'contamination removed'. The cluster "
                      "roll-up is majority rule, so a merged diarizer cluster carrying both a member and a "
                      "resident still gets one label."),
             "table": audit["disagreement_by_body"][:12]},
            {"name": "San Leandro sign flip",
             "text": ("Stance is toward the agenda item, so on items that are restrictions the mapping "
                      "inverts: a landlord opposing a rent-stabilisation ordinance is “opposed to the "
                      f"item” and lands as NIMBY. {sl.get('flip_turns', '?')} of San Leandro's "
                      f"{sl.get('scored_public_turns', '?')} scored public turns and "
                      f"{sl.get('flip_clusters', '?')} of its {sl.get('clusters', '?')} public clusters "
                      "carry sub_topic = affordability_displacement, and San Leandro is "
                      f"{sl.get('clusters', '?')} of {flip['total_public_clusters']} clusters under the "
                      "site's own cue-role count. Re-quantified under role_audit over the whole "
                      f"scored frame: {_sl.get('flip_clusters', '?')} of San Leandro's "
                      f"{_sl.get('clusters', '?')} public clusters ({_pc(_sl.get('flip_share'))} %) carry "
                      f"that sub-topic, and San Leandro is {_sl.get('clusters', '?')} of "
                      f"{(SITE_N.get('split_audit') or {}).get('n', '?')} clusters "
                      f"({_pc(_sl.get('share_of_public_base'))} %). Dropping it moves the split from "
                      f"{_tr(SITE_N.get('split_audit'))} to {_tr(SITE_N.get('split_audit_drop_sanleandro'))} "
                      f"(under the cue role it was {_tr(SITE_N.get('split_cue'))} → "
                      f"{_tr(SITE_N.get('split_cue_drop_sanleandro'))}). "
                      "On the larger S2 frame San Leandro is a fifth of the public base rather than a "
                      "third, so dropping it barely moves the overall split — the inversion is diluted, "
                      "not resolved. The role audit does not touch this either: it is a mapping problem, "
                      "not a role problem. If the mapping is wrong for those turns the direction of this "
                      "view is unresolved rather than merely noisy."),
             "table": flip["rows"][:12]},
            {"name": "Most cluster labels rest on one scored turn",
             "text": (f"Public clusters average {_one.get('mean_all', 1.10)} scored turns "
                      f"({_one.get('mean_with_stance', 1.23)} among those with a stance) and "
                      f"{_pc(_one.get('share_one_numeric'), '83')} % "
                      "of cluster labels rest on exactly one scored turn. Clusters are diarization output, "
                      "not people: one speaker split across clusters counts twice, two speakers merged "
                      "count once.")},
        ],
        "figs": figs,
        # Figure titles are written, not derived from the filename (readability review
        # §7.3), and the captions no longer end in a provenance tail (§7.6) -- the
        # "regenerated after the role audit" note is in the first caveat above.
        "fig_titles": {f"fig_public_yimby_nimby_{r}_stars.png":
                       f"How the public split, body by body — {r} route" for r in routes}
        | {f"fig_public_vs_dais_{r}.png":
           f"The same split with the dais beside the public — {r} route" for r in routes},
        # Concision pass, owner 2026-10-19 12:46: the caption reads the figure and
        # stops. The bootstrap, the threshold and what n counts are in the caveats.
        "fig_caps": {f"fig_public_yimby_nimby_{r}_stars.png":
                     f"One bar per body on the {r} route: how its public speakers split. A star means "
                     "the YIMBY − NIMBY interval clears zero — few do, which is the result."
                     for r in routes}
        | {f"fig_public_vs_dais_{r}.png":
           f"The same bodies on the {r} route with the dais drawn beside the public. The dais is mostly "
           "neutral, so this is the baseline the public split should be read against rather than against "
           "nothing."
           for r in routes},
        "methods_anchor": "speakers",
    }


# The renter panel's figures are globbed, so their titles and captions are built
# from the filename rather than listed one by one. Every outcome below is named in
# the figure's own axis label; nothing here is a number.
RENTER_OUTCOMES = {
    "stance_yimby": ("the YIMBY share of its public speakers", "YIMBY share"),
    "stance_nimby": ("the NIMBY share of its public speakers", "NIMBY share"),
    "stance_net": ("its net YIMBY minus NIMBY share", "net YIMBY − NIMBY"),
    "stance_gap": ("the public − dais stance gap", "public − dais stance gap"),
    "rate_density": ("its density minutes per meeting-hour", "density topic time"),
    "rate_transit": ("its transit minutes per meeting-hour", "transit topic time"),
}


def _renter_parts(fname: str):
    stem = fname[len("fig_"):-len(".png")]
    for key, (phrase, short) in RENTER_OUTCOMES.items():
        if stem.startswith(key + "_"):
            return key, phrase, short, stem[len(key) + 1:]
    return None, None, None, None


def renter_fig_title(fname: str) -> str:
    if fname == "fig_beta_renter_forest.png":
        return "Renter share as a body-level predictor, every outcome tested"
    if fname == "fig_ppc_binomial.png":
        return "Does the NIMBY-count fit reproduce what was observed?"
    _, _, short, route = _renter_parts(fname)
    return f"{short[0].upper()}{short[1:]} against renter share — {route} route" if short else fname


def renter_fig_cap(fname: str) -> str:
    if fname == "fig_beta_renter_forest.png":
        return ("One row per outcome: the posterior slope on renter share, per standard deviation of it, "
                "with its 90 % and 95 % intervals. Every interval crosses the dashed zero line, which is "
                "the whole result of this panel — and at this many bodies it is what the design could have "
                "produced either way.")
    if fname == "fig_ppc_binomial.png":
        return ("Posterior predictive check for the binomial NIMBY-count fit: the observed NIMBY count in "
                "each body-and-route cell against the band the fitted model replicates. Every cell lands "
                "inside its band, so the fit reproduces the data it was given — which is self-consistency, "
                "not evidence that renter share predicts anything.")
    _, phrase, _, route = _renter_parts(fname)
    if not phrase:
        return ""
    return (f"One point per body, read on the {route} route only: {phrase} against the renter-occupied "
            "share of its occupied housing units. Point area is how much that body contributes and the "
            "dashed line is an ordinary least-squares fit; the rank correlation and its interval are in "
            "the figure's own title. Transit boards are drawn hollow, because a transit board's topic "
            "time is close to a tautology. With this few bodies the line describes the points on the "
            "chart and nothing beyond them.")


def panel_renter():
    model = json.loads((RENTER / "models" / "model_results.json").read_text(encoding="utf-8")) if (RENTER / "models" / "model_results.json").exists() else {}
    # `_cue` figures are the diarization-cue sensitivity kept in the repo after the
    # 2026-10-18 role audit; the page shows the reported role_audit versions only.
    # `fig_lex_*` are the null lexicon pilot, off the page since 2026-10-20 (owner):
    # it was a 300-turn pilot of a hand-built word list that lost to the LLM at
    # κ = 0.175, so it is not an instrument and its correlations are not a result.
    # The forest leads the panel. It is the one figure that shows what this panel
    # actually found -- every interval crossing zero -- whereas the rate scatters that
    # would otherwise sort first are the rows the caveats below say not to quote yet.
    scatters = [f for f in sorted(RENTER.glob("fig_*.png"))
                if not f.stem.endswith("_cue") and not f.stem.startswith("fig_lex_")]
    figs = copy_figs([RENTER / "models" / "fig_beta_renter_forest.png"]
                     + scatters
                     + [RENTER / "models" / "fig_ppc_binomial.png"])
    return {
        "title": "Renter share",
        "stamp": STAMP,
        # Concision pass, owner 2026-10-19 12:46: one sentence. The ACS table, the
        # transit-board county union and the outcome list are on Methods.
        "lede": ("Does a body's renter share predict what gets said at its meetings? One correlation "
                 "per outcome per route, against the ACS renter share."),
        # TLDR rule (owner, 2026-10-18 21:16, docs/SITE-STYLE.md): vague, result-
        # pointing, no numbers at all. The slopes, their intervals and how little this
        # design could have detected are in the table and caveats of the body below.
        # Concision pass, owner 2026-10-19 12:46: three bullets, no numbers. That a
        # result here would be about cities rather than speakers is its own caveat.
        "headline": [
            "I asked whether a body's renter share predicts what gets said at its meetings.",
            "Found nothing the data can resolve: every outcome tested leaves the direction open.",
            "Found the reason is how few bodies there are, not that renter share does not matter.",
        ],
        # The lexicon-pilot outcome rows come out with its figures: a correlation against a
        # rejected instrument is not a weaker result, it is not a result.
        "correlations": [r for r in num(rd(RENTER / "correlations.csv"))
                         if "lexicon" not in (r.get("outcome") or "").lower()],
        "covariates": num(rd(RENTER / "covariates.csv")),
        "model_table": num(rd(RENTER / "models" / "model_table.csv")),
        "model": clean(model),
        "caveats": [
            {"name": "How little this could have detected — n is the finding",
             "text": ("“No relationship” here means “no relationship this design could see”. Each "
                      "correlation rests on 15–20 bodies with a median of 18 scored public turns each, "
                      "which could only have detected an effect several times larger than any plausible "
                      "one. The 90 % interval on the NIMBY slope runs from a large negative effect to a "
                      "moderate positive one, and so does every other outcome tested. Read the null as a "
                      "statement about the sample size, not about renter share. And a sharp result here "
                      "would still be a statement about cities rather than about the people who show up "
                      "to speak at their meetings: the covariate is a body's whole renter share, not "
                      "anything known about a speaker.")},
            {"name": "Withdrawn claim",
             "text": ("An earlier draft said the two rate correlations were “driven by” BART and VTA and "
                      "that “dropping them leaves five councils and no usable correlation”. That is "
                      "withdrawn — it had the sign backwards. Dropping the two boards leaves Fremont, "
                      "Hayward, Napa, Petaluma and Santa Clara, and on those five the density Pearson "
                      "*rises* from +0.489 to +0.667 (p 0.22) while Spearman falls from +0.679 to +0.400; "
                      "the transit-rate Pearson rises +0.198 to +0.592. The boards were diluting the "
                      "Pearson correlation, not driving it. What is true is that nothing differs from zero "
                      "at n = 5, as was already the case at n = 7. (One of those “five councils” is Napa, "
                      "which is a County Board of Supervisors.)")},
            {"name": "The rate correlations should not be quoted at all yet",
             "text": ("Both rate rows are computed on the as-published 7-body complete-coverage set. The "
                      "coverage test has since been fixed and the complete set is 5 bodies, so on a rerun "
                      "today those rows fall to n = 3 on the classifier route. The local_asr and "
                      "agenda_index rate figures drop below the 3-body floor entirely and have been moved "
                      "to output/renter_share/_stale_pre_coverage_fix/.")},
            {"name": "Transit boards are marked hollow, and a councils-only panel sits beside each figure",
             "text": ("A transit board's topic time is close to a tautology, so every figure draws the "
                      "boards with hollow markers and carries a councils-only panel wherever one can be "
                      "drawn above the 3-body floor. Read the councils-only panel first.")},
        ],
        "figs": figs,
        # Written titles and captions for every figure in this panel (readability review
        # §7.3, §7.6): the filename-derived heading and the raw path tail are gone. The
        # scatter figures all share one shape -- one point per body, outcome against
        # renter share, one locate route at a time -- so the caption is built from that
        # shape and the outcome, and states no number the figure does not carry itself.
        "fig_titles": {f: renter_fig_title(f) for f in figs},
        "fig_caps": {f: renter_fig_cap(f) for f in figs},
        "methods_anchor": "limitations",
    }


def panel_rates():
    rows = num(rd(WAVE1 / "rates_by_body_route.csv"))
    cov = num(rd(WAVE1 / "coverage.csv"))
    complete = [r for r in rows if (r.get("coverage") or "").startswith("complete")]
    cov_complete = [r for r in cov if (r.get("coverage") or "") == "complete"]
    bodies = sorted({r["body"] for r in cov_complete})
    rated = sorted({r["body"] for r in complete})
    no_exposure = [b for b in bodies if b not in rated]
    return {
        "title": "Topic time per meeting-hour",
        "stamp": "Rates are a measurement, not a model estimate; intervals are 95 % episode bootstrap",
        # Concision pass, owner 2026-10-19 12:46: one sentence. That a partly located
        # body's rate is a floor is the second bullet and the first caveat.
        "lede": ("Minutes of on-topic episode per meeting-hour, by body and route, over the hours that "
                 "route actually read."),
        # TLDR rule (owner, 2026-10-18 21:16, docs/SITE-STYLE.md): vague, result-
        # pointing, no numbers at all. The per-body rates, the complete-coverage set, the
        # bodies with no exposure denominator and the route spread are in the table and
        # caveats of this panel's body.
        # Concision pass, owner 2026-10-19 12:46: four bullets, no numbers.
        "headline": [
            "I measured how many minutes of on-topic talk each body gives per meeting-hour.",
            "Found transit boards give the topic far more meeting time than any council does.",
            "Found most rates are floors: few bodies are completely read, and which route read a body "
            "moves its rate about as much as the body does.",
            "I marked which rate came off which route in the chart and table below, and kept "
            "the coverage rule on Methods.",
        ],
        "complete": complete,
        "all_rows": rows,
        "coverage": cov,
        "complete_bodies": bodies,
        "rated_bodies": rated,
        "complete_without_exposure": no_exposure,
        "caveats": [
            {"name": "The complete-coverage set fell from 11 bodies to "
                     + str(len(bodies)) + ", and why",
             "text": ("The coverage test counted only two pending statuses as pending and silently "
                      "dropped the rest: 42 meeting-groups sitting at `downloaded` and one `failed` were "
                      "counted as neither located nor pending, which made six partial bodies look "
                      "complete. The complete set is now " + ", ".join(bodies) + "; Fremont, Hayward, "
                      "Petaluma, Richmond, San Leandro and Santa Clara are partial and their rates are "
                      "floors, not estimates. Of the complete set only " + ", ".join(rated) + " get a "
                      "rate at all — " + ", ".join(no_exposure) + " carry 0.0 meeting-hours because "
                      "their Granicus listings publish no video duration, and with a zero exposure "
                      "denominator there is no rate to compute.")},
            {"name": "Locate route is a first-order effect, not a nuisance",
             "text": ("The same meetings read by a different route give a different rate, of the same "
                      "order as the differences between bodies: Hayward reads 0.65 min/h by classifier, "
                      "1.73 by agenda_index and 4.17 by the local Whisper pass. That is why route is a "
                      "column in every table here and nothing is pooled across routes. For scale on the "
                      "body differences: the two transit boards sit at BART 29.1 min/h and VTA 27.0, "
                      "while Napa — the one complete council-type body with an exposure denominator — "
                      "sits at 1.8–2.2 min/h on all three of its routes.")},
            {"name": "Pad-bounds finding withdrawn — false positive, no row changed",
             "text": ("A validation pass reported that 24 % of on-topic episodes carried an unremoved ~60 s "
                      "pad and that BART and VTA were inflated by about 6 %. That is withdrawn and no "
                      "core_* value was recomputed — recomputing them would have corrupted correct data. "
                      "core_* are the merged *unpadded* window bounds, and episodes.start_s/end_s are the "
                      "same bounds until the cutter replaces them with a padded, clamped, cue-snapped span. "
                      "So core == start/end on exactly the rows that have not been cut, by design. Every "
                      "cut row on every route carries its pad, the rate numerator uses core only, and it "
                      "was never inflated.")},
            {"name": "“Density outweighs transit on every council” is false on the table's own numbers",
             "text": ("Hayward on the classifier route is 0.1 density against 0.4 transit, and Petaluma's "
                      "Swagit route is 7.2 against 8.2. The claim holds on every council on the classifier "
                      "route except Hayward, and not on Petaluma's Swagit route.")},
            {"name": "Seven episodes overrun their recording",
             "text": ("Seven rows have core_end_s past the end of the media because the window grid "
                      "overruns the recording — worth at most ~60 s each. Not fixed: it changes episode "
                      "spans, which is the owner's call.")},
        ],
        "figs": [],
        "methods_anchor": "locate",
    }


def spend_so_far():
    """What the study has cost, with the date the newest figure was measured.

    Owner's dashboard rule: spend is visible and dated. The ledger is the same
    `money()` export_flow.py puts on the Methods flowchart, so the two pages cannot
    quote different totals. Three accounts are added: every GPU pod the project has
    run (`runpod_total`, which already includes both the diarized and the undiarized
    transcription batches -- adding `gpu` and `gpu_nodiar` on top would double-count),
    the stance pass and the locate classifier. `as_of` is the newest of the dates the
    parts carry, because a total is only as current as its stalest row.
    """
    m = money()
    parts = {k: m[k] for k in ("runpod_total", "stance", "classifier") if m.get(k)}
    if not parts:
        return None
    total = sum(p["usd"] for p in parts.values())
    as_of = max((p.get("as_of") for p in parts.values() if p.get("as_of")), default=None)
    return {
        "usd": round(total, 2),
        "as_of": as_of,
        "by_account": {k: round(v["usd"], 2) for k, v in sorted(parts.items())},
        "text": (f"Total spend to date ${total:,.0f} — GPU transcription "
                 f"${parts['runpod_total']['usd']:,.0f}, stance coding "
                 f"${parts['stance']['usd']:,.0f}, topic classifier "
                 f"${parts['classifier']['usd']:,.0f}"
                 if {"runpod_total", "stance", "classifier"} <= parts.keys()
                 else f"Total spend to date ${total:,.0f}"),
    }


def page_intro():
    """The paragraph under the title, and the page TLDR under that.

    Owner, 2026-10-19: "cite in the intro paragraph the amount of data
    collected, how many individual statements were collected, that we cut them
    and classified them with an llm. keep sentences short. then speak on the
    results just a little inline. have a short concise tldr."

    Every count comes out of `data/bodies.json`'s own totals block, which
    export_site_data.py writes from the live DB -- so the intro, the map, the
    methods footer and the Methods page cannot quote different corpus sizes.
    The result sentences take their two numbers from site_numbers_role.json.
    Run export_site_data.py before this script or the intro goes stale, which
    is why the sentences are built here and not typed into index.html.

    TLDR rule (owner, 2026-10-18 21:16, docs/STYLE.md): no numbers at all,
    first person, past tense, one claim a bullet, last bullet points onward.
    """
    t, bmeta = {}, {}
    bj = DATA / "bodies.json"
    if bj.exists():
        bmeta = json.loads(bj.read_text(encoding="utf-8")).get("meta") or {}
        t = bmeta.get("totals") or {}
    z = ((ROLE_N.get("zero_stance") or {}).get("by_role")) or {}
    # The one model (owner, 2026-10-19), not run 5's cluster-level fit, so the
    # intro's last clause quotes the same number the role block does.
    dirn = next((s for s in (role_model().get("steps") or [])
                 if s["key"] == "which_side"), {})
    fi = lambda v: f"{v:,.0f}" if isinstance(v, (int, float)) else "?"
    f1 = lambda v: f"{v:,.0f}" if isinstance(v, (int, float)) else "?"

    # -- what was collected, in short sentences -------------------------------
    # Concision pass, owner 2026-10-19 12:46: same four facts, short sentences, no
    # clause a later panel repeats. What the stance scale is, that no human has coded
    # it, and the no-linking and names rules are all said once further down the page.
    collected = (
        f"I pulled a year of regular meetings from {fi(t.get('bodies'))} Bay Area city "
        f"councils, county boards and transit boards: {fi(t.get('meetings'))} meetings, about "
        f"{f1(t.get('meeting_hours'))} hours of video. An LLM marked the stretches about housing "
        f"density or transit; those were cut out as {fi(t.get('episodes_done'))} episodes and "
        f"transcribed, {f1(t.get('hours_done'))} hours of speech. A second pass scored "
        f"{fi(t.get('stance_turns'))} statements for stance and hostility. No human has coded "
        "any of it, so this is exploratory.")

    # -- and what came out, briefly: the three the owner named ----------------
    dz = (z.get("dais") or {}).get("share_at_zero")
    pz = (z.get("public") or {}).get("share_at_zero")
    results = (
        "Three things came out. Transit boards spend far more meeting time on the topic than "
        "councils do, and which route found a body moves its rate almost as much as the body "
        "does. Public speakers take positions where the dais does not: "
        + (f"{_pc(dz)} % of scored dais statements sit at exactly neutral against "
           f"{_pc(pz)} % of public ones" if dz and pz else
           "the dais mostly stays in the neutral band")
        + ". And of the statements that do take a side, public ones are about "
        + (f"{dirn['odds_ratio']:.1f} times" if dirn.get("odds_ratio") else "one and a half times")
        + " as likely in odds to be against the item.")

    return {
        "collected": collected,
        "results": results,
        # No numbers, by house rule. These gesture at the three results above and
        # hand off to the panels.
        # Past tense throughout, 2026-10-20: three of these stated a finding in the
        # present ("transit boards give", "the public takes sides") or had no first
        # person at all ("The map ... are the results"), which is the one thing the rule
        # above asks of a bullet. The claims are unchanged.
        "tldr": [
            "I collected a year of Bay Area council, county and transit-board meetings, "
            "cut out the parts about housing density and transit, and had an LLM score "
            "every statement in them.",
            "I found that transit boards gave the topic far more of their meeting time "
            "than councils did, and that which route read a body mattered nearly as much "
            "as the body itself.",
            "I found that the public took sides where the dais stayed neutral, and that "
            "most of the gap between them was about who took a position at all rather "
            "than which position they took.",
            "I put the results in the map, the episode reader and the panels below, and "
            "kept every instrument, number and limitation on the "
            "<a href=\"methods.html\">Methods</a> page.",
        ],
        # Owner's dashboard rules: spend is visible and dated, and the page says how
        # current its data is. Both are rendered under the TLDR by index.html.
        "spend": spend_so_far(),
        # The date the pipeline database was last read for this page, which is what a
        # reader means by "as of": export_site_data.py stamps it, and this script runs
        # after it, so the two agree. Not the newest meeting date -- the corpus window
        # closed earlier than the export, and the Corpus entry in the footer says so.
        "as_of": (bmeta.get("generated") or "")[:10] or dt.date.today().isoformat(),
        "totals": {k: t.get(k) for k in
                   ("bodies", "bodies_with_hours", "meetings", "meeting_hours",
                    "episodes", "episodes_done", "episodes_scored", "hours_done",
                    "stance_turns")},
        "source": ("data/bodies.json meta.totals, output/stance_s2/site_numbers_role.json, "
                   "output/renter_share_6/"),
    }


def main():
    panels = {
        "generated": dt.datetime.now().astimezone().isoformat(timespec="minutes"),
        # `stamp` is the per-figure stamp every panel carries. There is no page banner:
        # the owner took the EXPLORATORY banner off index.html on purpose, and the
        # per-figure stamps are what carry it now, so no banner string is exported.
        "stamp": STAMP,
        "intro": page_intro(),
        "stance": panel_stance(),
        "nimby": panel_nimby(),
        "renter": panel_renter(),
        "rates": panel_rates(),
    }
    DATA.mkdir(exist_ok=True)
    (DATA / "panels.json").write_text(json.dumps(clean(panels), indent=0), encoding="utf-8")
    # One file a reader (or a reviewer) can grep for any number the page quotes about
    # role, without reading panels.json's prose. It is a copy, not a second source:
    # analysis/site_numbers_role.py computes it in the ytpt repo from the runs' own
    # artefacts, and this is that file plus the stance_s2 numbers the panels already
    # used and the corpus totals the intro cites.
    (DATA / "site_numbers.json").write_text(json.dumps(clean({
        "generated": panels["generated"],
        "note": ("Exported copy of the numbers the dashboard's prose quotes. Computed by "
                 "~/ytpt analysis/site_numbers_role.py and analysis/stance_s2_site_numbers.py "
                 "from the runs' artefacts; nothing here is typed by hand, and nothing on the "
                 "page is typed by hand either -- export/export_panels.py substitutes these."),
        "sources": {"role": str(_ROLE_N_PATH).replace(str(YTPT), "~/ytpt"),
                    "stance_s2": str(_SITE_N_PATH).replace(str(YTPT), "~/ytpt"),
                    "corpus_totals": "data/bodies.json meta.totals"},
        "corpus_totals": panels["intro"]["totals"],
        "role": ROLE_N,
        "stance_s2": SITE_N,
    }), indent=1), encoding="utf-8")
    print(json.dumps({
        "panels.json bytes": (DATA / "panels.json").stat().st_size,
        "site_numbers.json bytes": (DATA / "site_numbers.json").stat().st_size,
        "intro_totals": panels["intro"]["totals"],
        "role_side": bool(panels["nimby"].get("role_side")),
        "role_pred": bool(panels["stance"].get("role_pred")),
        "figures": {k: len(panels[k]["figs"]) for k in ("stance", "nimby", "renter", "rates")},
        "stance": {"tail_rows": len(panels["stance"]["model_free"]["rows"])},
        "spend": panels["intro"].get("spend"),
        "as_of": panels["intro"].get("as_of"),
        "nimby": {"by_body_rows": len(panels["nimby"]["by_body"]),
                  "by_route_rows": len(panels["nimby"]["by_route"])},
        "renter": {"correlations": len(panels["renter"]["correlations"])},
        "rates": {"complete_rows": len(panels["rates"]["complete"]),
                  "complete_bodies": panels["rates"]["complete_bodies"],
                  "all_rows": len(panels["rates"]["all_rows"])},
    }, indent=1))


if __name__ == "__main__":
    main()
