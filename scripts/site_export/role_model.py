#!/usr/bin/env python
"""The one model the dashboard reports the role gap from (READ-ONLY).

Owner, 2026-10-19: "lets keep that lme4 model then as the one we use in the
dashboard. drop the others from the final explanation but keep it in the repo."

So: one estimator, two steps, and nothing else quoted on the page. The model is
the nested turn-level mixed logistic regression of plan/RENTER-SHARE-6-R.md --
one row per transcript turn, random intercepts for the episode, the meeting and
the body it sits in, fixed effects for speaker role (public against dais),
renter share, their interaction and locate route. Two outcomes, same
specification, same frame:

  took a side at all   every stance-bearing turn; did it take a side rather
                       than sit in the neutral band
                       (analysis/r/export_anyside.py + analysis/r/turn_anyside.R)
  which side, given a side
                       the turns that took one; NIMBY rather than YIMBY
                       (analysis/r/export_for_r.py + analysis/r/turn_profile.R)

Every number below is read out of output/renter_share_6/, never retyped. The
other fits of runs 2 to 5 -- the Bayesian horseshoe, the GEE, the custom
Gauss-Hermite GLMM, the beta-binomial -- stay in the repo and are not quoted on
the site; the one thing said about them is how far they disagree with this fit,
and that number is computed here from comparison_r.csv rather than asserted.

Intervals are profile-likelihood 95 %, which is what the scripts write; where a
fit is singular in `body` the profile comes from the identical non-singular
reparameterisation, and `profiled_levels` records which. Where the profile
bounds a term at neither end the Wald interval is substituted instead, and
`ci95_kind` says which kind every interval is.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

YTPT = Path("~/ytpt").expanduser()
RS6 = YTPT / "output" / "renter_share_6"

#: Profile-interval rows are keyed by lme4's parameter names.
_ROLE = "role_code"
_RENTER = "z_renter"
_INTER = "role_x_renter"


def _rd(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _jl(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _f(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isinf(f) or math.isnan(f) else f


def _or(beta):
    return None if beta is None else math.exp(beta)


def _share(rows, role, key="y"):
    sub = [r for r in rows if r.get("role") == role]
    if not sub:
        return None
    return sum(float(r[key]) for r in sub) / len(sub)


def _step(name, key, turns_csv, fixed_csv, prof_csv, meta, outcome, event,
          scripts):
    """One outcome of the one model, assembled from its own artefacts."""
    fixed = {r["parameter"]: r for r in _rd(RS6 / fixed_csv)}
    prof = {r["parameter"]: r for r in _rd(RS6 / prof_csv)}
    if _ROLE not in fixed or _ROLE not in prof:
        return None
    turns = _rd(RS6 / "data" / turns_csv)
    vc = {r["grp"]: _f(r["sdcor"]) for r in _rd(RS6 / meta["varcomp"])}
    jm = _jl(RS6 / meta["json"]) if meta.get("json") else {}

    def par(nm):
        fx, pr = fixed.get(nm) or {}, prof.get(nm) or {}
        beta, se = _f(fx.get("est")), _f(fx.get("se"))
        lo, hi = _f(pr.get("prof_lo")), _f(pr.get("prof_hi"))
        # A profile that bounds at neither end publishes nothing: the which_side
        # renter term came back -Inf to Inf on the 2026-10-20 refit. Fall back to
        # the Wald interval from the same fit's standard error and carry the kind,
        # so no page can show a likelihood label over a number the likelihood did
        # not produce. One-sided profiles keep their bounded end, which is how
        # plan/FINAL-2026-10-20.md §4a quotes role x renter.
        kind = "profile"
        if lo is None and hi is None and beta is not None and se is not None:
            lo, hi, kind = beta - 1.96 * se, beta + 1.96 * se, "wald"
        elif lo is None or hi is None:
            kind = "unbounded" if lo is None and hi is None else "profile_one_sided"
        return {"beta": beta, "se": se, "p": _f(fx.get("p")),
                "ci95": [lo, hi], "ci95_kind": kind, "odds_ratio": _or(beta),
                "or_ci95": [_or(lo), _or(hi)]}

    role = par(_ROLE)
    return {
        "key": key,
        "name": name,
        "outcome": outcome,
        "event": event,
        **role,
        "renter": par(_RENTER),
        "interaction": par(_INTER),
        "n_turns": len(turns) or None,
        "n_events": int(sum(float(r["y"]) for r in turns)) if turns else None,
        "rate_public": _share(turns, "public"),
        "rate_dais": _share(turns, "dais"),
        "n_episodes": jm.get("n_episodes") or len({r["episode_id"] for r in turns}) or None,
        "n_meetings": jm.get("n_meetings") or len({r["meeting_id"] for r in turns}) or None,
        "n_bodies": jm.get("n_bodies") or len({r["body"] for r in turns}) or None,
        "sd_episode": vc.get("episode_id"),
        "sd_meeting": vc.get("meeting_id"),
        "sd_body": vc.get("body"),
        "profiled_levels": jm.get("profiled_levels", 2),
        "singular_body": bool(jm.get("singular_3level")),
        "devtol": jm.get("profile_devtol"),
        "scripts": scripts,
    }


def _agreement():
    """How far the fits kept in the repo disagree with this one, in log-odds.

    plan/RENTER-SHARE-6-R.md section 8: the custom Gauss-Hermite GLMM of run 5
    and lme4::glmer agree to a few ten-thousandths on every coefficient and
    both denominators. Recomputed here from comparison_r.csv so the Methods
    line cannot drift from the comparison it cites.
    """
    rows = _rd(RS6 / "comparison_r.csv")
    want = {}
    for r in rows:
        k = (r["denominator"], r["parameter"])
        if r["fit"] == "Python GH-GLMM":
            want.setdefault(k, {})["py"] = _f(r["est"])
        elif r["fit"].startswith("lme4::glmer (Wald"):
            want.setdefault(k, {})["r"] = _f(r["est"])
    diffs = [abs(v["py"] - v["r"]) for v in want.values()
             if v.get("py") is not None and v.get("r") is not None]
    return {"max_abs_logodds": max(diffs) if diffs else None,
            "n_pairs": len(diffs)} if diffs else {}


def role_model() -> dict:
    """The whole block, or {"available": False} if run 6 has not been run."""
    steps = [
        _step("Took a side at all", "any_side", "turns_anyside.csv",
              "anyside_fixed.csv", "anyside_profile_ci.csv",
              {"varcomp": "anyside_varcomp.csv", "json": "anyside_profile.json"},
              outcome="every stance-bearing turn",
              event="took a side rather than staying in the neutral band",
              scripts=["analysis/r/export_anyside.py",
                       "analysis/r/turn_anyside.R"]),
        _step("Which side, given a side", "which_side", "turns.csv",
              "turn_fixed_2level.csv", "turn_profile_ci.csv",
              {"varcomp": "turn_varcomp.csv", "json": "turn_profile.json"},
              outcome="the turns that took a side",
              event="came out against the item rather than for it",
              scripts=["analysis/r/export_for_r.py",
                       "analysis/r/turn_profile.R"]),
    ]
    steps = [s for s in steps if s]
    if not any(s["key"] == "which_side" for s in steps):
        return {"available": False, "source": "output/renter_share_6/"}

    # Transit commute is the one covariate besides role that clears 95 % in
    # this run, and it is a body-level quantity: the turn model does not carry
    # it, so its number is the glmer fit on the body-level rows of
    # RENTER-SHARE-6-R.md section 3. Same package, same estimator, profile
    # interval; it is named as a different fit on the page.
    prim = {r["parameter"]: r for r in _rd(RS6 / "glmer_primary_profile.csv")}
    pwald = {r["parameter"]: r for r in _rd(RS6 / "glmer_primary_wald.csv")}
    tc, tw = prim.get("transit_commute") or {}, pwald.get("transit_commute") or {}
    transit = ({"beta": _f(tw.get("est")),
                "ci95": [_f(tc.get("prof_lo")), _f(tc.get("prof_hi"))],
                "scope": "body-level rows, NIMBY out of NIMBY + YIMBY"}
               if tc else None)

    return {
        "available": True,
        "steps": steps,
        "transit_commute": transit,
        "agreement": _agreement(),
        "random_effects": "episode inside meeting inside body",
        "scripts": sorted({s for st in steps for s in st["scripts"]}),
        "doc": "plan/RENTER-SHARE-6-R.md",
        "source": "output/renter_share_6/",
    }


if __name__ == "__main__":
    print(json.dumps(role_model(), indent=1))
