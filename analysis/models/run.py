#!/usr/bin/env python3
"""
run.py -- `make models`: build both estimands end to end and write
output/analysis/models/{tables/*.csv, figs/*.png, summary.json}.

Estimand 1's primary fit is the stratified table (load.cell_table_pooled,
pool=True) with locate route as a fixed-effect stratum -- explicit, stamped
POOLED, never silent. The per-route sweep (load.cell_table + fit_per_route)
is the sensitivity: one fit per route, never pooled, used for the contrast
figure's per-route rows and nothing else.

Estimand 2 runs on the real `turn_stance` table when it exists, otherwise on
the VADER stand-in (load.standin_turns / bin_standin), capped at
load.CAP_H hours per body (SHOWCASE.md 1b). Swept per role_source level
(facet guard) and per outcome (stance, hostility).
"""
import argparse, sys, time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from analysis.models import load as L, topic_time as TT, role_gap as RG, plots as P  # noqa: E402

SCOPE_T1 = "topic time: every meeting-hour-exposed body x month x family x route cell, current corpus"
SCOPE_T2 = "role gap: turns >= 100 words, capped at %.0f h per body" % L.CAP_H


def _write(df, path):
    if df is None or len(df) == 0:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


# -------------------------------------------------------------- estimand 1 ----
def topic_time_stage(out_dir, state=None, draws=1000, tune=1000, chains=4, sensitivity=True):
    out_dir = Path(out_dir)
    tables, figs = out_dir / "tables", out_dir / "figs"
    tables.mkdir(parents=True, exist_ok=True)
    figs.mkdir(parents=True, exist_ok=True)
    st = L.open_state(state)
    try:
        cells_pooled = L.cell_table_pooled(st, outcome="located", both="credit", pool=True)
        cells_raw = L.cell_table(st, outcome="located", both="credit")
    finally:
        st.close()
    report = {"n_cells_pooled": int(len(cells_pooled)), "n_cells_raw": int(len(cells_raw)),
              "routes": sorted(cells_raw["route"].unique().tolist())}

    t0 = time.time()
    main_fit = TT.fit(cells_pooled, draws=draws, tune=tune, chains=chains)
    summ = TT.summarise(main_fit)
    _write(summ["bodies"], tables / "POOLED_topic_time_bodies.csv")
    _write(summ["contrast"], tables / "POOLED_topic_time_contrast.csv")
    _write(summ["months"], tables / "POOLED_topic_time_months.csv")
    _write(summ["routes"], tables / "POOLED_topic_time_routes.csv")
    _write(summ["hyper"], tables / "POOLED_topic_time_hyper.csv")
    _write(summ["ppc"], tables / "POOLED_topic_time_ppc.csv")

    per_route_rows = []
    skipped_routes = {}
    if sensitivity:
        fits_pr, skipped_routes = TT.fit_per_route(cells_raw, draws=draws, tune=tune, chains=chains)
        for route, f in fits_pr.items():
            c = TT.contrast_table(f)
            row = c[c["contrast"].str.contains(" / ")]
            if len(row):
                per_route_rows.append((f"{route} (per-route)", row.iloc[0]))
            _write(TT.body_table(f), tables / f"topic_time_bodies_route-{route}.csv")

    P.shrinkage(summ["bodies"], figs / "topic_time_shrinkage.png", SCOPE_T1, True)
    P.body_forest(summ["bodies"], figs / "topic_time_body_forest.png", SCOPE_T1, True)
    if len(per_route_rows):
        P.contrast(summ["contrast"], per_route_rows, figs / "topic_time_contrast.png", SCOPE_T1, True)
    if len(summ["months"]):
        P.months(summ["months"], figs / "topic_time_months.png", SCOPE_T1, True)
    P.ppc(summ["ppc"], figs / "topic_time_ppc.png", SCOPE_T1, True, likelihood=main_fit["likelihood"])

    report.update({
        "seconds": round(time.time() - t0, 1),
        "diagnostics": main_fit["diagnostics"],
        "skipped_routes": skipped_routes,
        "n_route_fits": len(per_route_rows),
    })
    return report


# -------------------------------------------------------------- estimand 2 ----
def _load_turns(corpus, state):
    """(turns, instrument, is_standin); the standin path is uncapped here --
    role_gap_stage applies load.cap_by_body once, after this returns."""
    if L.turn_stance_exists(state):
        st = L.open_state(state)
        try:
            df = L.read_turn_stance(st)
        finally:
            st.close()
        return df, "stance pass", False
    return L.bin_standin(L.standin_turns(corpus)), L.STANDIN, True


def role_gap_stage(out_dir, corpus=None, state=None, draws=1000, tune=1000, chains=4):
    out_dir = Path(out_dir)
    tables, figs = out_dir / "tables", out_dir / "figs"
    tables.mkdir(parents=True, exist_ok=True)
    figs.mkdir(parents=True, exist_ok=True)
    corpus = corpus or L.DEFAULT_CORPUS
    if not Path(corpus).exists() and not L.turn_stance_exists(state):
        L.build_corpus(db_path=corpus, state=state)

    turns, instrument, is_standin = _load_turns(corpus, state)
    body_hours_kept = {}
    if is_standin:
        ep_hours = L.episode_hours(corpus)
        turns, body_hours_kept = L.cap_by_body(turns, ep_hours)

    t0 = time.time()
    fits, skipped = RG.sweep(turns, facet="role_source", outcomes=("stance", "hostility"),
                              draws=draws, tune=tune, chains=chains)
    diagnostics = {}
    for (level, outcome), f in fits.items():
        summ = RG.summarise(f)
        tag = f"{outcome}_{level}"
        _write(summ["gap"], tables / f"role_gap_gap_{tag}.csv")
        _write(summ["coef"], tables / f"role_gap_coef_{tag}.csv")
        _write(summ["bodies"], tables / f"role_gap_bodies_{tag}.csv")
        _write(summ["ppc"], tables / f"role_gap_ppc_{tag}.csv")
        P.role_gap_forest(summ["gap"], figs / f"role_gap_forest_{tag}.png", instrument, SCOPE_T2, outcome=outcome)
        P.role_gap_ppc(summ["ppc"], figs / f"role_gap_ppc_{tag}.png", instrument, SCOPE_T2, outcome=outcome,
                       levels=f["levels"])
        diagnostics[tag] = f["diagnostics"]

    return {"seconds": round(time.time() - t0, 1), "instrument": instrument, "is_standin": is_standin,
            "n_turns_total": int(len(turns)), "body_hours_kept": body_hours_kept,
            "fitted": sorted(f"{lev}/{oc}" for lev, oc in fits.keys()),
            "skipped": {f"{lev}/{oc}": v for (lev, oc), v in skipped.items()},
            "diagnostics": diagnostics}


# -------------------------------------------------------------------- main ----
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(ROOT / "output" / "analysis" / "models"))
    ap.add_argument("--corpus", default=str(L.DEFAULT_CORPUS))
    ap.add_argument("--state", default=None)
    ap.add_argument("--draws", type=int, default=1000)
    ap.add_argument("--tune", type=int, default=1000)
    ap.add_argument("--chains", type=int, default=4)
    ap.add_argument("--no-sensitivity", action="store_true")
    a = ap.parse_args(argv)
    out = Path(a.out)
    t0 = time.time()
    t1 = topic_time_stage(out, state=a.state, draws=a.draws, tune=a.tune, chains=a.chains,
                          sensitivity=not a.no_sensitivity)
    t2 = role_gap_stage(out, corpus=a.corpus, state=a.state, draws=a.draws, tune=a.tune, chains=a.chains)
    summary = {"topic_time": t1, "role_gap": t2, "total_seconds": round(time.time() - t0, 1),
               "preliminary": True, "freeze_date": "2026-10-24",
               "note": "role_gap instrument is %s" % t2["instrument"]}
    L.write_json(out / "summary.json", summary)
    print("topic_time: %.1fs, max_rhat=%.3f, divergences=%d" %
          (t1["seconds"], t1["diagnostics"]["max_rhat"], t1["diagnostics"]["divergences"]))
    print("role_gap:   %.1fs, instrument=%s, fitted=%s" % (t2["seconds"], t2["instrument"], t2["fitted"]))
    print("total:      %.1fs" % summary["total_seconds"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
