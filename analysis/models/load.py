#!/usr/bin/env python3
"""
load.py -- data construction for the two showcase models. Read-only against
state.sqlite; writes nothing but the corpus file it is asked for.

ESTIMAND 1 (topic time). The unit is a CELL: body x month x family x locate
route. Its exposure is the meeting-hours that route READ in that body-month; its
outcome is the whole minutes of located episodes that route found there.

  exposure   every kind='meeting' row whose status says a locate route finished
             with it ('cut' or 'no_episodes'), with a known duration. 'captioned'
             and 'discovered' meetings have not been read by any route, so their
             hours are not exposure: a zero there would be missingness wearing a
             number. All kind='meeting' rows are regular meetings by construction
             (pipeline/discover.py select_regular). Sausalito and SamTrans have no
             duration on 42 meetings (A0-WIDE-RESULTS 4) and drop out of the rate
             model with a note, not a zero.
  route      a meeting's routes are the union of: the `method` of any episodes
             row it has (located or mention), 'classifier' if classifier_calls
             read it, and the route its caption kind implies (CAPTIONS_ROUTE) --
             so a 'no_episodes' meeting with no rows still has the route that
             read it and found nothing. One meeting can be exposure for two
             routes (Santa Clara: classifier + agenda_index); each route's cell
             carries it, and the stratified table counts it once.
  outcome    'located': every episodes row with an about-label (or the NULL
             label of an agenda_index row) and status != 'failed'. Bounds
             (core_start_s/core_end_s, else start_s/end_s) come from the locate
             step, not from transcription, so transcribed-or-not does not change
             the minutes -- using only `done` would make the outcome depend on
             which GPU batch has run. 'done' is the sensitivity. Minutes are
             rounded to whole minutes for the count likelihood; the classifier's
             bounds are caption-window resolution, coarser than that.
  both       an about_both episode is time about density AND time about
             transit. The default credits its minutes to both families; the
             alternatives ('exclude', or a third family level 'both') are
             sensitivities. Double-crediting makes the two family cells of one
             body-month share minutes, which the body x family random effect
             absorbs; it does not change the body ranking.

FACET. Route is an instrument (analysis/facets.py). cell_table() is per route
and nothing in it mixes routes. pool_routes() builds the stratified table (one
row per body x month x family, route = the dominant one as a fixed-effect
stratum) and REFUSES unless pool=True; its output carries a `pooled` column so a
figure drawn from it can be stamped. The per-route fits are the per-facet
result; the stratified fit is the explicitly pooled one.

ESTIMAND 2 (role gap). read_turn_stance() reads the `turn_stance` table the
stance pass (pipeline/stance.py, WP A) will write. standin_turns() builds the
same columns from VADER compound per turn (analysis/a1_sentiment.py's loader)
restricted to turns >= MIN_WORDS words (A1-RESULTS 1: VADER is mostly a length
measure below that) and binned into five quantile levels; every row carries
`instrument='STAND-IN: VADER, not stance'` and every figure prints it.
"""
import json, os, sqlite3, sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "analysis"))
sys.path.insert(0, str(ROOT / "pipeline"))

SCOPE_STATUSES = ("cut", "no_episodes")
FAMILIES = ("density", "transit")
ROUTES = ("classifier", "agenda_index", "local_asr", "swagit_transcript")
CAPTIONS_ROUTE = {"granicus": "classifier", "auto": "classifier", "manual": "classifier",
                  "asr_local": "local_asr", "swagit": "swagit_transcript", "none": "agenda_index"}
MIN_WORDS = 100          # A1-RESULTS 1: the length control VADER needs
STANDIN = "STAND-IN: VADER, not stance"
LEVELS = (-2, -1, 0, 1, 2)
CAP_H = 8.0              # SHOWCASE.md 1b: role-gap sample capped at 8 h per body
DEFAULT_CORPUS = Path(os.path.expanduser("~/.local/share/ytpt/corpus_models.sqlite"))


class FacetError(SystemExit):
    """Raised when a caller would pool locate routes without saying so."""


def state_path(explicit=None):
    import state
    return Path(explicit) if explicit else Path(state.db_path())


def open_state(path=None):
    p = state_path(path)
    con = sqlite3.connect(f"file:{p}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    return con


# ------------------------------------------------------------- estimand 1 ----
def meeting_routes(st):
    """{video_id: set(route)} for every in-scope meeting, see module docstring."""
    routes = defaultdict(set)
    for r in st.execute("SELECT DISTINCT video_id, COALESCE(method, 'unknown') m FROM episodes"):
        routes[r["video_id"]].add(r["m"])
    try:
        for r in st.execute("SELECT DISTINCT video_id FROM classifier_calls"):
            routes[r["video_id"]].add("classifier")
    except sqlite3.Error:
        pass
    for r in st.execute("SELECT video_id, captions FROM videos WHERE kind = 'meeting'"):
        if r["captions"] in CAPTIONS_ROUTE:
            routes[r["video_id"]].add(CAPTIONS_ROUTE[r["captions"]])
    return routes


def scope_meetings(st):
    """DataFrame of meetings that are exposure, and the ones dropped for no duration."""
    rows = [dict(r) for r in st.execute(
        "SELECT video_id, speaker AS body, month, duration_s FROM videos "
        "WHERE kind = 'meeting' AND status IN (%s)" % ",".join("?" * len(SCOPE_STATUSES)), SCOPE_STATUSES)]
    df = pd.DataFrame(rows, columns=["video_id", "body", "month", "duration_s"])
    ok = df["duration_s"].notna() & (df["duration_s"] > 0)
    return df[ok].copy(), df[~ok]


def located_episodes(st, outcome="located"):
    """One row per episode that counts toward the numerator, with its minutes."""
    assert outcome in ("located", "done")
    rows = [dict(r) for r in st.execute(
        "SELECT episode_id, video_id, COALESCE(method, 'unknown') AS route, tags AS family, label, status, "
        "COALESCE(core_start_s, start_s) AS s0, COALESCE(core_end_s, end_s) AS s1 FROM episodes")]
    df = pd.DataFrame(rows, columns=["episode_id", "video_id", "route", "family", "label", "status", "s0", "s1"])
    df = df[(df["label"].isna() | (df["label"] != "mention_only")) & (df["status"] != "failed")]
    if outcome == "done":
        df = df[df["status"] == "done"]
    df = df[df["s0"].notna() & df["s1"].notna()].copy()
    df["minutes"] = (df["s1"] - df["s0"]).clip(lower=0) / 60.0
    return df


def cell_table(st, outcome="located", both="credit"):
    """body x month x family x route cells. Columns: body, month, family, route,
    exposure_h, n_meetings, n_episodes, minutes, y (rounded minutes)."""
    assert both in ("credit", "exclude", "third")
    meets, _ = scope_meetings(st)
    routes = meeting_routes(st)
    eps = located_episodes(st, outcome)
    fams = list(FAMILIES) + (["both"] if both == "third" else [])
    # exposure per (body, month, route)
    exp_rows = []
    for m in meets.itertuples(index=False):
        for r in routes.get(m.video_id, set()):
            exp_rows.append((m.body, m.month, r, m.duration_s / 3600.0, m.video_id))
    ex = pd.DataFrame(exp_rows, columns=["body", "month", "route", "hours", "video_id"])
    if ex.empty:
        return pd.DataFrame(columns=["body", "month", "family", "route", "exposure_h", "n_meetings", "n_episodes", "minutes", "y"])
    exposure = ex.groupby(["body", "month", "route"]).agg(exposure_h=("hours", "sum"), n_meetings=("video_id", "nunique")).reset_index()
    # numerator per (body, month, route, family) -- only episodes whose meeting is exposure
    meet_info = meets.set_index("video_id")[["body", "month"]]
    eps = eps.join(meet_info, on="video_id", how="inner")
    num_rows = []
    for e in eps.itertuples(index=False):
        if e.family == "both":
            targets = list(FAMILIES) if both == "credit" else (["both"] if both == "third" else [])
        else:
            targets = [e.family] if e.family in FAMILIES else []
        for f in targets:
            num_rows.append((e.body, e.month, e.route, f, e.minutes, e.episode_id))
    num = pd.DataFrame(num_rows, columns=["body", "month", "route", "family", "minutes", "episode_id"])
    num = num.groupby(["body", "month", "route", "family"]).agg(minutes=("minutes", "sum"), n_episodes=("episode_id", "nunique")).reset_index()
    grid = exposure.merge(pd.DataFrame({"family": fams}), how="cross")
    cells = grid.merge(num, on=["body", "month", "route", "family"], how="left")
    cells["minutes"] = cells["minutes"].fillna(0.0)
    cells["n_episodes"] = cells["n_episodes"].fillna(0).astype(int)
    cells["y"] = np.rint(cells["minutes"]).astype(int)
    cells["pooled"] = False
    return cells.sort_values(["route", "body", "month", "family"]).reset_index(drop=True)


def pool_routes(cells, pool=False):
    """The stratified table: one row per body x month x family, exposure = hours
    of DISTINCT meetings (a meeting read by two routes is one meeting), y summed
    over routes, `route` = the dominant route (most minutes, else most exposure)
    as a fixed-effect stratum. Refused unless pool=True (analysis/facets.py)."""
    if cells["route"].nunique() <= 1:
        out = cells.copy(); out["pooled"] = False
        return out
    if not pool:
        raise FacetError(
            "refusing to pool locate routes: this cell table spans %s.\n"
            "  Routes are instruments, not variables (analysis/facets.py, plan/T2-PLAN.md).\n"
            "  Fit per route (topic_time.fit_per_route), or pass pool=True to build the\n"
            "  stratified table -- every figure and file from it is stamped POOLED." % "/".join(sorted(cells["route"].unique())))
    # exposure of distinct meetings needs the meeting ids; cells only has hours per
    # route. Re-derive from the max over routes when a body-month's routes read the
    # same meetings (classifier + agenda_index at Santa Clara read the same
    # meetings; local_asr bodies have one route). The exact distinct-meeting union
    # is computed by cell_table_pooled() when a state connection is available.
    g = cells.groupby(["body", "month", "family"])
    out = g.agg(exposure_h=("exposure_h", "max"), n_meetings=("n_meetings", "max"),
                n_episodes=("n_episodes", "sum"), minutes=("minutes", "sum")).reset_index()
    dom = (cells.sort_values(["minutes", "exposure_h"], ascending=False)
           .drop_duplicates(["body", "month", "family"])[["body", "month", "family", "route"]])
    out = out.merge(dom, on=["body", "month", "family"], how="left")
    out["y"] = np.rint(out["minutes"]).astype(int)
    out["pooled"] = True
    return out.sort_values(["body", "month", "family"]).reset_index(drop=True)


def cell_table_pooled(st, outcome="located", both="credit", pool=False):
    """pool_routes() with the exact distinct-meeting exposure from state.sqlite."""
    cells = cell_table(st, outcome, both)
    out = pool_routes(cells, pool=pool)
    if not out["pooled"].any():
        return out
    meets, _ = scope_meetings(st)
    routes = meeting_routes(st)
    meets = meets[meets["video_id"].map(lambda v: bool(routes.get(v)))]
    ex = meets.groupby(["body", "month"]).agg(exposure_h=("duration_s", lambda s: s.sum() / 3600.0),
                                               n_meetings=("video_id", "nunique")).reset_index()
    out = out.drop(columns=["exposure_h", "n_meetings"]).merge(ex, on=["body", "month"], how="left")
    return out


def raw_rates(cells, by=("body",)):
    """Pooled-over-cells crude rate, minutes per meeting-hour, for the shrinkage plot."""
    by = list(by)
    num = cells.groupby(by).agg(minutes=("minutes", "sum"), n_cells=("y", "size")).reset_index()
    # exposure is per (body, month, route); the family rows of one cell share it
    keys = [c for c in ("body", "month", "route") if c in cells.columns]
    ex = cells.drop_duplicates(keys + [c for c in by if c not in keys]).groupby(by)["exposure_h"].sum().reset_index()
    g = num.merge(ex, on=by, how="left")
    g["raw_rate"] = g["minutes"] / g["exposure_h"].replace(0, np.nan)
    return g


# ------------------------------------------------------------- the corpus ----
def build_corpus(db_path=DEFAULT_CORPUS, qa_csv=None, state=None):
    """The QA-gated transcript corpus into `db_path` (default
    ~/.local/share/ytpt/corpus_models.sqlite), the load_episodes --wide pattern:
    every body, every route, episodes with an OK/WARN row in the QA report."""
    import load_episodes as le, load_db
    qa_csv = Path(qa_csv or (ROOT / "output" / "qa" / "latest.csv"))
    if not qa_csv.exists():
        alt = Path("~/ytpt/output/qa/latest.csv").expanduser()   # worktrees do not carry output/
        qa_csv = alt if alt.exists() else qa_csv
    qa = le.select_qa_gate(qa_csv)
    db_path = Path(db_path); db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()
    con, n_ep, n_seg, n_words, sources, n_meet = le.load(state_path(state), str(db_path), load_db.collapse_loops, qa_gate=qa)
    con.close()
    return {"db": str(db_path), "qa_csv": str(qa_csv), "qa_gate": len(qa), "episodes": n_ep,
            "segments": n_seg, "words": n_words, "role_sources": sources, "meetings": n_meet}


# ------------------------------------------------------------- estimand 2 ----
TURN_COLS = ["episode_id", "cluster", "turn_idx", "stance_median", "hostility_median",
             "role", "body", "family", "words"]


def read_turn_stance(con):
    """The real table (pipeline/stance.py). Returns the TURN_COLS frame plus
    `instrument='stance pass'`; rows with NULL stance are kept (hostility may
    still be scored) and dropped per model."""
    try:
        con.execute("SELECT 1 FROM turn_stance LIMIT 1")
    except sqlite3.Error:
        return None
    df = pd.read_sql_query("SELECT %s FROM turn_stance" % ", ".join(TURN_COLS), con)
    df["instrument"] = "stance pass"
    return df


def turn_stance_exists(path=None):
    try:
        st = open_state(path)
    except sqlite3.Error:
        return False
    try:
        st.execute("SELECT 1 FROM turn_stance LIMIT 1")
        return True
    except sqlite3.Error:
        return False
    finally:
        st.close()


def standin_turns(corpus_db, frames_path=None, hostility_path=None):
    """VADER compound per turn from a1_sentiment.load_turns, pooled over the
    instrument facets ON PURPOSE (role_source and method are carried as columns
    and the model stratifies on them). Returns the raw turn frame."""
    import a0_terms as at, a1_sentiment as a1
    frames = at.load_frames(frames_path or ROOT / "plan" / "frames.yaml")
    hostility = at.load_frames(hostility_path or ROOT / "plan" / "hostility.txt")
    rows, _scope = a1.load_turns(str(corpus_db), frames, hostility, pool=True)
    df = pd.DataFrame(rows)
    return df.rename(columns={"speaker": "cluster", "turn": "turn_idx"})


def bin_standin(df, min_words=MIN_WORDS, levels=LEVELS):
    """Turns >= min_words, compound -> 5 quantile levels mapped onto -2..2,
    hostility hits -> 0..4 (capped). Every row carries the STAND-IN label."""
    d = df[df["words"] >= min_words].copy()
    if d.empty:
        return d
    k = len(levels)
    ranks = d["compound"].rank(method="first")
    d["stance_median"] = pd.qcut(ranks, k, labels=list(levels)).astype(int)
    d["hostility_median"] = d["hostility_hits"].clip(upper=4).astype(int)
    d["instrument"] = STANDIN
    keep = TURN_COLS + ["instrument", "compound", "role_source", "method", "talk_s"]
    return d[[c for c in keep if c in d.columns]].reset_index(drop=True)


def episode_hours(corpus_db):
    """{episode_id: hours} from the corpus `episodes` table (talk seconds)."""
    con = sqlite3.connect(f"file:{corpus_db}?mode=ro", uri=True)
    try:
        return {r[0]: (r[1] or 0.0) / 3600.0 for r in con.execute("SELECT episode_id, duration_s FROM episodes")}
    finally:
        con.close()


def cap_by_body(turns, ep_hours, cap_h=CAP_H, seed=20261017):
    """SHOWCASE.md 1b: draw episodes at random within each body until the cap
    is reached, so no body contributes more than ~cap_h hours of turns. Returns
    (capped turns, {body: hours kept})."""
    rng = np.random.default_rng(seed)
    keep, kept_h = set(), {}
    for body, g in turns.groupby("body"):
        eps = list(g["episode_id"].unique())
        rng.shuffle(eps)
        tot = 0.0
        for e in eps:
            h = ep_hours.get(e, 0.0)
            if tot and tot + h > cap_h:
                continue
            keep.add(e); tot += h
        kept_h[body] = round(tot, 2)
    return turns[turns["episode_id"].isin(keep)].reset_index(drop=True), kept_h


def write_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, sort_keys=True, default=_json_default) + "\n")


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.ndarray,)):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    return str(o)
