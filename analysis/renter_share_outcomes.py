#!/usr/bin/env python3
"""
renter_share_outcomes.py -- per-body outcomes for plan/RENTER-SHARE.md,
one table per locate route, never pooled (analysis/facets.py).

Outcomes, all per body x route:
  A. public stance shares (primary, from the S1 census of turn_stance):
       NIMBY = public turns with stance_median <= -0.5
       YIMBY = public turns with stance_median >= +0.5
     denominator = public turns with a numeric stance in density/both-family
     episodes, the YIMBY/NIMBY mapping and threshold of
     STANCE-S1-RESULTS.md 7b. Unit here is the TURN, not the speaker cluster:
     the cluster table (nimby_view) has >=10 clusters in only 8+2+1+1 bodies,
     and the turn table reaches more bodies at the same floor. Cluster-level
     shares are carried alongside for comparison.
  A2. public - dais NIMBY-share gap, same frame, model-free.
  B. density and transit located minutes per meeting-hour, from
     output/wave1/rates_by_body_route.csv; complete-coverage bodies only are
     marked `complete`, the rest are floors and are flagged.
  C. the lexicon pilot's per-body public NIMBY/YIMBY share
     (output/nimby_pilot/public_share_by_body_<route>.csv), the weaker
     fallback instrument, carried for a sanity comparison only.

Correlation: Spearman and Pearson of each outcome against renter share over
bodies within a route, with 2,000-rep body-bootstrap percentile CIs.
Figures: one scatter per outcome per route, labelled by body, point area
proportional to n (turns or meeting-hours), EXPLORATORY stamp.

The role column is switchable and defaults to the **audited** role (rule ra-1,
state.sqlite `role_audit`), which is the reported one as of 2026-10-18
(plan/ROLE-AUDIT-2026-10-18.md 7). Every stance outcome here is a share of the
public bucket, so all of A, A2 and C move with it; the rate outcomes (B) do
not. `--role-col cue` reproduces the published cue run under a `_cue` suffix.

Reads only: corpus.sqlite, state.sqlite (read-only URIs), output/wave1/,
output/nimby_pilot/, output/stance_s1/, output/renter_share/covariates.csv.
Writes output/renter_share/{outcomes_by_body_route,correlations}<sfx>.csv,
outcomes<sfx>.json, fig_*_<route><sfx>.png. No spend, nothing committed.

    .venv/bin/python analysis/renter_share_outcomes.py                 # audit (reported)
    .venv/bin/python analysis/renter_share_outcomes.py --role-col cue  # _cue sensitivity
"""
import argparse
import csv
import json
import pathlib
import sqlite3
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "output" / "renter_share"
_SHARE = pathlib.Path("~/.local/share/ytpt").expanduser()
STATE = f"file:{_SHARE}/state.sqlite?mode=ro"
CORPUS = f"file:{_SHARE}/corpus.sqlite?mode=ro"

STAMP = "EXPLORATORY — stance κ gate not passed (plan/SHOWCASE.md 4a)"
THRESH = 0.5           # STANCE-S1-RESULTS 7b
MIN_N = 5              # primary floor on public turns with a numeric stance
MIN_N_STRICT = 10      # sensitivity floor
SEED = 20261018
REPS = 2000

ROUTES = ["classifier", "local_asr", "agenda_index", "swagit_transcript"]

#: Which role column supplies `role`. The **audited** role (rule ra-1,
#: scripts/role_audit.py, state.sqlite `role_audit`) is the reported one as of
#: 2026-10-18 -- plan/ROLE-AUDIT-2026-10-18.md 7. The diarization-cue role
#: mislabels 20-39 % of the adjudicable public-bucket turns as public when the
#: speaker's own words say `elected` or `staff`, and this file's stance
#: outcomes are all public-bucket shares, so they move with the audit. The cue
#: run is kept as a `_cue` sensitivity. Set by main().
ROLE_COL = "audit"
SFX = ""
SUFFIX = {"audit": "", "cue": "_cue"}


# ---------------------------------------------------------------- covariates
def covariates():
    out = {}
    for r in csv.DictReader(open(OUT / "covariates.csv")):
        out[r["body"]] = dict(
            renter_share=float(r["renter_share"]),
            med_hh_income=float(r["med_hh_income"]),
            pop_density=float(r["pop_density_per_sq_mi"]),
            geo_kind=r["geo_kind"], census_names=r["census_names"])
    return out


# ------------------------------------------------------------------ outcomes
def audit_roles(st):
    """(episode_id, cluster, turn_idx) -> cluster_role_audit, the ra-1 cluster
    roll-up. Turns with no row (or self_id unknown throughout the cluster) are
    absent and the caller keeps the cue role, as the rule specifies."""
    return {(e, c, t): r for e, c, t, r in st.execute(
        "select episode_id,cluster,turn_idx,cluster_role_audit from role_audit")
        if r in ("public", "dais")}


def stance_counts():
    """(route, body, role) -> dict of NIMBY/YIMBY/neutral/n counts, turn unit."""
    co = sqlite3.connect(CORPUS, uri=True)
    ep = {r[0]: (r[1], r[2]) for r in
          co.execute("select episode_id,method,family from episodes")}
    st = sqlite3.connect(STATE, uri=True)
    ra = audit_roles(st) if ROLE_COL == "audit" else {}
    cnt = defaultdict(lambda: dict(NIMBY=0, YIMBY=0, neutral=0, n=0,
                                   na=0, episodes=set()))
    for epid, clu, tix, body, role, stance, sm in st.execute(
            "select episode_id,cluster,turn_idx,body,role,stance,stance_median "
            "from turn_stance"):
        m = ep.get(epid)
        if not m:
            continue
        route, fam = m
        role = ra.get((epid, clu, tix), role)
        if fam not in ("density", "both") or role not in ("public", "dais"):
            continue
        c = cnt[(route, body, role)]
        c["episodes"].add(epid)
        if stance == "not_applicable" or sm is None:
            c["na"] += 1
            continue
        c["n"] += 1
        if sm <= -THRESH:
            c["NIMBY"] += 1
        elif sm >= THRESH:
            c["YIMBY"] += 1
        else:
            c["neutral"] += 1
    for c in cnt.values():
        c["episodes"] = len(c["episodes"])
    return cnt


def stance_turns():
    """Flat per-turn rows for the bootstrap and the GLM (public only)."""
    co = sqlite3.connect(CORPUS, uri=True)
    ep = {r[0]: (r[1], r[2]) for r in
          co.execute("select episode_id,method,family from episodes")}
    st = sqlite3.connect(STATE, uri=True)
    ra = audit_roles(st) if ROLE_COL == "audit" else {}
    rows = []
    for epid, clu, tix, body, role, stance, sm in st.execute(
            "select episode_id,cluster,turn_idx,body,role,stance,stance_median "
            "from turn_stance"):
        m = ep.get(epid)
        if not m:
            continue
        route, fam = m
        role = ra.get((epid, clu, tix), role)
        if fam not in ("density", "both") or role not in ("public", "dais"):
            continue
        if stance == "not_applicable" or sm is None:
            continue
        rows.append(dict(episode=epid, route=route, body=body, role=role,
                         nimby=int(sm <= -THRESH), yimby=int(sm >= THRESH)))
    return rows


def cluster_shares():
    """(route, body) -> cluster-level public YIMBY/NIMBY share from 7b, under
    the same role column (the 7b view writes `_cue` for the cue run)."""
    out = {}
    p = ROOT / "output" / "stance_s1" / "nimby_view" / \
        f"public_dais_shares_by_body_route{SFX}.csv"
    if not p.exists():
        print(f"note: {p.name} missing; cl_YIMBY/cl_NIMBY left blank "
              f"(run analysis/stance_s1_nimby_view.py --role-col {ROLE_COL})")
        return out
    for r in csv.DictReader(open(p)):
        if r["role"] != "public":
            continue
        out[(r["route"], r["body"])] = dict(
            cl_YIMBY=float(r["YIMBY_share"]), cl_NIMBY=float(r["NIMBY_share"]),
            cl_n=int(r["n_clusters"]))
    return out


def rates():
    """(route, body) -> density/transit min per meeting-h + coverage."""
    out = defaultdict(dict)
    for r in csv.DictReader(open(ROOT / "output" / "wave1" /
                                 "rates_by_body_route.csv")):
        k = (r["route"], r["body"])
        out[k]["coverage"] = r["coverage"]
        out[k]["meeting_hours"] = float(r["meeting_hours"] or 0)
        out[k]["meetings"] = int(r["meetings"] or 0)
        if r["family"] in ("density", "transit"):
            out[k][f"rate_{r['family']}"] = float(r["rate_min_per_h"] or 0)
            out[k][f"rate_{r['family']}_lo"] = float(r["ci95_lo"] or 0)
            out[k][f"rate_{r['family']}_hi"] = float(r["ci95_hi"] or 0)
    return out


def lexicon_pilot():
    out = {}
    for route in ROUTES:
        p = ROOT / "output" / "nimby_pilot" / f"public_share_by_body_{route}.csv"
        if not p.exists():
            continue
        for r in csv.DictReader(open(p)):
            out[(route, r["body"])] = dict(
                lex_YIMBY=float(r["YIMBY_share"]), lex_NIMBY=float(r["NIMBY_share"]),
                lex_n=int(r["n_turns"]))
    return out


# --------------------------------------------------------------- correlation
def boot_corr(x, y, w=None, reps=REPS, seed=SEED):
    """Spearman/Pearson with a body-resampling percentile CI."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    n = len(x)
    if n < 4:
        return None
    rho = stats.spearmanr(x, y).statistic
    r = stats.pearsonr(x, y).statistic
    rng = np.random.default_rng(seed)
    bs_rho, bs_r = [], []
    for _ in range(reps):
        i = rng.integers(0, n, n)
        if len(set(x[i])) < 3 or len(set(y[i])) < 3:
            continue
        bs_rho.append(stats.spearmanr(x[i], y[i]).statistic)
        bs_r.append(stats.pearsonr(x[i], y[i]).statistic)
    bs_rho = np.array([v for v in bs_rho if np.isfinite(v)])
    bs_r = np.array([v for v in bs_r if np.isfinite(v)])
    q = lambda a, p: float(np.percentile(a, p)) if len(a) else float("nan")
    return dict(n=n, spearman=float(rho),
                spearman_lo90=q(bs_rho, 5), spearman_hi90=q(bs_rho, 95),
                spearman_lo95=q(bs_rho, 2.5), spearman_hi95=q(bs_rho, 97.5),
                spearman_p=float(stats.spearmanr(x, y).pvalue),
                pearson=float(r),
                pearson_lo90=q(bs_r, 5), pearson_hi90=q(bs_r, 95),
                pearson_lo95=q(bs_r, 2.5), pearson_hi95=q(bs_r, 97.5),
                pearson_p=float(stats.pearsonr(x, y).pvalue),
                boot_ok=int(len(bs_rho)))


#: the three bodies whose geography is a union of whole counties -- the transit
#: boards. Marked hollow on every figure and given their own councils-only
#: panel since 2026-10-18: on the rate outcomes they are the two or three
#: points that carry the spread, and dropping them *raises* the density
#: Pearson (+0.489 -> +0.667), so a reader has to be able to see both fits
#: (plan/VALIDATION-2026-10-18.md 5).
BOARD_GEO = "county_union"


def is_board(r):
    return r.get("geo_kind") == BOARD_GEO


def _panel(ax, rows, xk, yk, nk, ylabel, title, smax):
    """One scatter panel: boards hollow, councils filled, OLS over what is drawn."""
    for sub, kw in ((False, dict(color="#2b6cb0", edgecolor="#1a365d", alpha=0.6)),
                    (True, dict(facecolors="none", edgecolor="#742a2a", linewidths=1.6))):
        pts = [r for r in rows if is_board(r) == sub]
        if not pts:
            continue
        nn = np.array([max(r[nk], 1) for r in pts], float)
        ax.scatter([r[xk] for r in pts], [r[yk] for r in pts],
                   s=30 + 320 * (nn / smax), zorder=3,
                   label=("transit board (county union)" if sub else "council / county board"), **kw)
    for r in rows:
        ax.annotate(r["body"], (r[xk], r[yk]), textcoords="offset points",
                    xytext=(7, 4), fontsize=8, color="#2d3748")
    x = [r[xk] for r in rows]
    y = [r[yk] for r in rows]
    if len(x) >= 3:
        b, a = np.polyfit(x, y, 1)
        xs = np.linspace(min(x), max(x), 50)
        ax.plot(xs, a + b * xs, color="#c05621", lw=1.4, ls="--", zorder=2,
                label=f"OLS slope {b:+.2f}")
    ax.legend(fontsize=7, loc="best")
    c = boot_corr(x, y) if len(x) >= 3 else None
    sub = (f"n = {len(x)} · Spearman {c['spearman']:+.2f} "
           f"(90 % {c['spearman_lo90']:+.2f}, {c['spearman_hi90']:+.2f}) · "
           f"Pearson {c['pearson']:+.2f} (p {c['pearson_p']:.2f})"
           if c else f"n = {len(x)} · too few for a correlation")
    ax.set_xlabel("renter-occupied share of occupied housing units\n"
                  "(ACS 2019–2023 5-year, B25003)")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title}\n{sub}", fontsize=9)
    ax.grid(alpha=0.25, zorder=0)
    return c


def scatter(rows, xk, yk, nk, route, outcome, ylabel, fn, note=""):
    councils = [r for r in rows if not is_board(r)]
    boards = [r for r in rows if is_board(r)]
    smax = max(max(r[nk], 1) for r in rows)
    two = bool(boards) and len(councils) >= 3
    fig, axes = plt.subplots(1, 2 if two else 1, figsize=((12.4, 5.6) if two else (7.6, 5.6)))
    axes = list(np.atleast_1d(axes))
    c = _panel(axes[0], rows, xk, yk, nk, ylabel, f"{outcome} — locate route: {route}", smax)
    c_councils = None
    if two:
        c_councils = _panel(axes[1], councils, xk, yk, nk, ylabel,
                            f"councils only — {len(boards)} transit board(s) dropped", smax)
    foot = STAMP + ("  ·  " + note if note else "")
    fig.text(0.01, 0.015, foot + "  ·  point area ∝ " +
             ("public turns" if "turn" in nk or nk == "n" else "meeting-hours") +
             "  ·  hollow = transit board (county union)" +
             ("  ·  right panel: same outcome without the boards" if two else ""),
             fontsize=7, color="#742a2a")
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    fig.savefig(OUT / fn, dpi=140)
    plt.close(fig)
    if c is not None and c_councils is not None:
        c = dict(c, councils_only_n=len(councils),
                 councils_only_spearman=c_councils["spearman"],
                 councils_only_pearson=c_councils["pearson"],
                 councils_only_pearson_p=c_councils["pearson_p"])
    return fn, c


def main(argv=None):
    global ROLE_COL, SFX
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--role-col", choices=tuple(SUFFIX), default="audit",
                    help="audit = the ra-1 cluster roll-up (reported, "
                         "ROLE-AUDIT-2026-10-18 7); cue = the published "
                         "diarization-cue role, written with a _cue suffix")
    ap.add_argument("--suffix", default=None)
    a = ap.parse_args(argv)
    ROLE_COL = a.role_col
    SFX = a.suffix if a.suffix is not None else SUFFIX[ROLE_COL]
    print(f"role column '{ROLE_COL}' (reported: audit); output suffix '{SFX}'")
    OUT.mkdir(parents=True, exist_ok=True)
    cov = covariates()
    sc = stance_counts()
    cl = cluster_shares()
    rt = rates()
    lex = lexicon_pilot()

    table = []
    for (route, body, role), c in sorted(sc.items()):
        if role != "public" or body not in cov:
            continue
        dais = sc.get((route, body, "dais"), dict(NIMBY=0, n=0))
        row = dict(route=route, body=body, **cov[body],
                   public_turns_numeric=c["n"], public_turns_na=c["na"],
                   public_episodes=c["episodes"],
                   nimby_count=c["NIMBY"], yimby_count=c["YIMBY"],
                   neutral_count=c["neutral"],
                   nimby_share=(c["NIMBY"] / c["n"]) if c["n"] else None,
                   yimby_share=(c["YIMBY"] / c["n"]) if c["n"] else None,
                   net_yimby=((c["YIMBY"] - c["NIMBY"]) / c["n"]) if c["n"] else None,
                   dais_turns_numeric=dais["n"],
                   dais_nimby_share=(dais["NIMBY"] / dais["n"]) if dais["n"] else None)
        row["nimby_gap_public_minus_dais"] = (
            row["nimby_share"] - row["dais_nimby_share"]
            if row["nimby_share"] is not None and row["dais_nimby_share"] is not None
            else None)
        row.update(cl.get((route, body), dict(cl_YIMBY=None, cl_NIMBY=None, cl_n=0)))
        row.update(lex.get((route, body), dict(lex_YIMBY=None, lex_NIMBY=None, lex_n=0)))
        r = rt.get((route, body), {})
        row.update(coverage=r.get("coverage"), meeting_hours=r.get("meeting_hours"),
                   meetings=r.get("meetings"),
                   rate_density=r.get("rate_density"), rate_transit=r.get("rate_transit"))
        table.append(row)

    # rate-only rows: bodies with a rate cell but no public stance turns
    have = {(r["route"], r["body"]) for r in table}
    for (route, body), r in sorted(rt.items()):
        if (route, body) in have or body not in cov:
            continue
        table.append(dict(route=route, body=body, **cov[body],
                          public_turns_numeric=0, public_turns_na=0,
                          public_episodes=0, nimby_count=0, yimby_count=0,
                          neutral_count=0, nimby_share=None, yimby_share=None,
                          net_yimby=None, dais_turns_numeric=0,
                          dais_nimby_share=None, nimby_gap_public_minus_dais=None,
                          cl_YIMBY=None, cl_NIMBY=None, cl_n=0,
                          lex_YIMBY=None, lex_NIMBY=None, lex_n=0,
                          coverage=r.get("coverage"),
                          meeting_hours=r.get("meeting_hours"),
                          meetings=r.get("meetings"),
                          rate_density=r.get("rate_density"),
                          rate_transit=r.get("rate_transit")))

    cols = list(table[0].keys())
    with open(OUT / f"outcomes_by_body_route{SFX}.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in sorted(table, key=lambda r: (r["route"], r["body"])):
            w.writerow(r)

    # ---- correlations + figures, per route, per outcome
    cors, figs = [], []
    specs = [
        ("nimby_share", "public_turns_numeric", "public NIMBY share",
         "NIMBY share of public turns with a stance", "stance_nimby",
         lambda r: r["nimby_share"] is not None and r["public_turns_numeric"] >= MIN_N),
        ("yimby_share", "public_turns_numeric", "public YIMBY share",
         "YIMBY share of public turns with a stance", "stance_yimby",
         lambda r: r["yimby_share"] is not None and r["public_turns_numeric"] >= MIN_N),
        ("net_yimby", "public_turns_numeric", "public YIMBY − NIMBY",
         "YIMBY − NIMBY share of public turns", "stance_net",
         lambda r: r["net_yimby"] is not None and r["public_turns_numeric"] >= MIN_N),
        ("nimby_gap_public_minus_dais", "public_turns_numeric",
         "public − dais NIMBY-share gap", "public NIMBY share − dais NIMBY share",
         "stance_gap",
         lambda r: r["nimby_gap_public_minus_dais"] is not None
         and r["public_turns_numeric"] >= MIN_N and r["dais_turns_numeric"] >= MIN_N),
        ("rate_density", "meeting_hours", "density minutes per meeting-hour",
         "located density min per meeting-h", "rate_density",
         lambda r: r.get("rate_density") is not None and r.get("coverage") == "complete"),
        ("rate_transit", "meeting_hours", "transit minutes per meeting-hour",
         "located transit min per meeting-h", "rate_transit",
         lambda r: r.get("rate_transit") is not None and r.get("coverage") == "complete"),
        ("lex_NIMBY", "lex_n", "lexicon-pilot NIMBY share (weaker instrument)",
         "lexicon NIMBY share of public turns", "lex_nimby",
         lambda r: r.get("lex_NIMBY") is not None and r.get("lex_n", 0) >= MIN_N),
    ]
    for yk, nk, outcome, ylabel, slug, keep in specs:
        for route in ROUTES:
            rows = [r for r in table if r["route"] == route and keep(r)]
            if len(rows) < 3:
                cors.append(dict(route=route, outcome=outcome, n=len(rows),
                                 note="too few bodies"))
                continue
            note = ("complete-coverage bodies only" if slug.startswith("rate")
                    else f"floor {MIN_N} public turns with a stance")
            fn, c = scatter(rows, "renter_share", yk, nk, route, outcome,
                            ylabel, f"fig_{slug}_{route}{SFX}.png", note)
            figs.append(str((OUT / fn).relative_to(ROOT)))
            rec = dict(route=route, outcome=outcome, bodies=len(rows),
                       bodies_list=";".join(r["body"] for r in rows), **(c or {}))
            # controls: partial correlation of the outcome on renter share
            # holding income and density (only where n permits)
            if len(rows) >= 6:
                X = np.column_stack([[r["renter_share"] for r in rows],
                                     [np.log(r["med_hh_income"]) for r in rows],
                                     [np.log(r["pop_density"]) for r in rows],
                                     np.ones(len(rows))])
                yv = np.array([r[yk] for r in rows], float)
                beta, *_ = np.linalg.lstsq(X, yv, rcond=None)
                rec["ols3_renter_slope"] = float(beta[0])
                res = yv - X @ beta
                dof = len(rows) - X.shape[1]
                if dof > 0:
                    s2 = res @ res / dof
                    se = float(np.sqrt(s2 * np.linalg.pinv(X.T @ X)[0, 0]))
                    rec["ols3_renter_se"] = se
                    rec["ols3_renter_lo95"] = float(beta[0] - 1.96 * se)
                    rec["ols3_renter_hi95"] = float(beta[0] + 1.96 * se)
            cors.append(rec)
            # strict-floor sensitivity
            if not slug.startswith("rate"):
                srows = [r for r in rows if r["public_turns_numeric"] >= MIN_N_STRICT]
                if len(srows) >= 4:
                    sc2 = boot_corr([r["renter_share"] for r in srows],
                                    [r[yk] for r in srows])
                    cors.append(dict(route=route,
                                     outcome=outcome + f" (floor {MIN_N_STRICT})",
                                     bodies=len(srows),
                                     bodies_list=";".join(r["body"] for r in srows),
                                     **(sc2 or {})))

    keys = sorted({k for c in cors for k in c})
    order = ["route", "outcome", "bodies", "n", "spearman", "spearman_lo90",
             "spearman_hi90", "spearman_lo95", "spearman_hi95", "spearman_p",
             "pearson", "pearson_lo90", "pearson_hi90", "pearson_lo95",
             "pearson_hi95", "pearson_p", "ols3_renter_slope", "ols3_renter_se",
             "ols3_renter_lo95", "ols3_renter_hi95", "boot_ok", "note",
             "bodies_list"]
    fields = [k for k in order if k in keys] + [k for k in keys if k not in order]
    with open(OUT / f"correlations{SFX}.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for c in cors:
            w.writerow(c)

    (OUT / f"outcomes{SFX}.json").write_text(json.dumps(dict(
        stamp=STAMP, threshold=THRESH, floor=MIN_N, floor_strict=MIN_N_STRICT,
        seed=SEED, bootstrap_reps=REPS, role_col=ROLE_COL,
        role_col_note=("ra-1 cluster roll-up, plan/ROLE-AUDIT-2026-10-18.md 7"
                       if ROLE_COL == "audit" else "diarization-cue role (sensitivity)"),
        unit="turn (public turns with a numeric stance), density/both families",
        acs="see output/renter_share/acs.json",
        figures=figs, correlations=cors), indent=2, default=str))

    for c in cors:
        if "spearman" in c:
            print(f"{c['route']:18s} {c['outcome'][:42]:42s} n={c['bodies']:2d} "
                  f"rho={c['spearman']:+.3f} "
                  f"[{c['spearman_lo90']:+.2f},{c['spearman_hi90']:+.2f}] "
                  f"r={c['pearson']:+.3f} p={c['pearson_p']:.3f}")
        else:
            print(f"{c['route']:18s} {c['outcome'][:42]:42s} n={c['n'] if 'n' in c else c.get('bodies')} -- {c.get('note')}")
    print(f"\n{len(figs)} figures, {len(table)} body x route rows -> {OUT}")


if __name__ == "__main__":
    main()
