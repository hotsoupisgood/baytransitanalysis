#!/usr/bin/env python3
"""
wave1_rates.py -- topic time per meeting-hour, per body x locate route, with
meeting-cluster bootstrap intervals, for the bodies whose locate coverage is
complete (analysis/wave1_coverage.py). $0, state.sqlite only, no network.

WHAT IS ESTIMATED: for a body b and locate route r, the located on-topic
minutes per regular-meeting hour,

    rate_{b,r,f} = sum over located episodes of route r, family f, of core
                   minutes  /  sum over the body's located regular meetings of
                   meeting hours                                    [min / h]

Every located episode counts (transcribed or not -- locating is what makes the
numerator, transcription only adds text), mention_only rows do not, failed rows
do not. The denominator is the body's whole located exposure, the same rule as
analysis/load_episodes.py meeting_exposure: a meeting is exposure for every
route that read it, so only the numerator splits by route. Family 'density'
means tags density or both; 'transit' means transit or both; 'any' is every
on-topic episode once.

WHY ONLY COMPLETE-COVERAGE BODIES: a body with meetings still waiting for a
caption track has an unknown numerator on those hours; its rate is a floor and
is not comparable with a complete body's. Those bodies get a row flagged
'partial (floor)' and are kept out of any cross-body statement.

UNCERTAINTY: a nonparametric cluster bootstrap that resamples whole meetings
within body (2,000 replicates, percentile 2.5 / 97.5), i.e. the sampling unit
is the meeting, not the episode and not the word. This is a per-body interval
with no pooling across bodies; the partial-pooling version is estimand 1 in
analysis/models (negative-binomial, body intercepts, route as a stratum) and
is the number to report once it has been refit on this set.

    .venv/bin/python analysis/wave1_rates.py --out-dir output/wave1
"""
import argparse, csv, os, sqlite3, sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "analysis"))
# one definition of located / pending / rank, shared with the coverage table so
# the two cannot disagree about what a status means (2026-10-18 validation)
from wave1_coverage import LOCATED, status_rank     # noqa: E402
FAMILIES = {"density": ("density", "both"), "transit": ("transit", "both"), "any": ("density", "transit", "both")}


def state_db_path(explicit=None):
    return Path(os.path.expanduser(explicit or os.environ.get("STATE_DB_PATH")
                                   or "~/.local/share/ytpt/state.sqlite"))


def read(state):
    st = sqlite3.connect(f"file:{state}?mode=ro", uri=True)
    st.row_factory = sqlite3.Row
    meet = {}   # meeting_id -> {body, status, hours, video_ids}
    vid2meet = {}
    for r in st.execute("SELECT video_id, speaker, status, duration_s, meeting_id FROM videos WHERE kind = 'meeting'"):
        key = r["meeting_id"] or r["video_id"]
        vid2meet[r["video_id"]] = key
        m = meet.setdefault(key, {"body": r["speaker"], "status": r["status"], "hours": 0.0})
        if status_rank(r["status"]) > status_rank(m["status"]):
            m["status"] = r["status"]
        m["hours"] = max(m["hours"], (r["duration_s"] or 0.0) / 3600)
    # numerator: (meeting, route, family) -> core minutes; read: (meeting, route)
    # -> this route produced at least one row (any status, mention included) for
    # the meeting, or the classifier was called on it. The route-read set is the
    # SENSITIVITY denominator (what that instrument actually looked at); a
    # meeting an agenda index read and found nothing in leaves no row, so this
    # set is a slight undercount for agenda_index and its rate a slight
    # overstatement.
    num = defaultdict(float)
    n_eps = defaultdict(int)
    read_by = defaultdict(set)
    for r in st.execute("SELECT DISTINCT video_id FROM classifier_calls"):
        key = vid2meet.get(r["video_id"])
        if key is not None:
            read_by[(key, "classifier")].add(r["video_id"])
    for r in st.execute("SELECT video_id, method, tags, label, status, start_s, end_s, core_start_s, core_end_s FROM episodes"):
        key = vid2meet.get(r["video_id"])
        if key is None:
            continue
        route = r["method"] or "unknown"
        read_by[(key, route)].add(r["video_id"])
        if r["label"] == "mention_only" or r["status"] == "failed":
            continue
        # core_* are the merged UNPADDED window bounds (classify.merge_episodes);
        # start_s/end_s are those same bounds until the cutter replaces them with
        # the padded, clamped, cue-snapped clip bounds (episodes.clip_bounds). So
        # core == start/end on any row that has not been cut, and that is correct,
        # not an unremoved pad: the numerator here only ever uses core. Checked
        # against classifier_calls.start_s/end_s on 2026-10-18 (validation §1
        # third bullet, a false positive).
        core = ((r["core_end_s"] or r["end_s"]) - (r["core_start_s"] or r["start_s"])) / 60
        for fam, tags in FAMILIES.items():
            if r["tags"] in tags:
                num[(key, route, fam)] += core
                n_eps[(key, route, fam)] += 1
    st.close()
    return meet, num, n_eps, read_by


def bootstrap(hours, mins, n_boot, rng):
    """rate replicates for one body: resample meetings with replacement."""
    hours = np.asarray(hours); mins = np.asarray(mins); n = len(hours)
    if n < 2 or hours.sum() <= 0:
        return None
    idx = rng.integers(0, n, size=(n_boot, n))
    h = hours[idx].sum(axis=1); m = mins[idx].sum(axis=1)
    ok = h > 0
    return m[ok] / h[ok]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--state", default=None)
    ap.add_argument("--coverage", default=str(ROOT / "output" / "wave1" / "coverage.csv"))
    ap.add_argument("--out-dir", default=str(ROOT / "output" / "wave1"))
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args(argv)
    state = state_db_path(a.state)
    if not state.exists():
        sys.exit(f"no state database at {state}")
    cov = {r["body"]: r["coverage"] for r in csv.DictReader(open(a.coverage, encoding="utf-8"))}
    meet, num, n_eps, read_by = read(state)
    rng = np.random.default_rng(a.seed)

    by_body = defaultdict(list)
    for key, m in meet.items():
        if m["status"] in LOCATED and m["hours"] > 0:
            by_body[m["body"]].append(key)
    rows = []
    for body in sorted(by_body):
        keys = by_body[body]
        hours = [meet[k]["hours"] for k in keys]
        routes = sorted({r for (k, r, f) in num if k in set(keys)})
        for route in routes:
            rkeys = [k for k in keys if (k, route) in read_by]
            rhours = [meet[k]["hours"] for k in rkeys]
            for fam in FAMILIES:
                mins = [num.get((k, route, fam), 0.0) for k in keys]
                eps = sum(n_eps.get((k, route, fam), 0) for k in keys)
                rate = sum(mins) / sum(hours)
                reps = bootstrap(hours, mins, a.n_boot, rng)
                lo, hi = (np.percentile(reps, [2.5, 97.5]) if reps is not None else (float("nan"), float("nan")))
                rmins = [num.get((k, route, fam), 0.0) for k in rkeys]
                rrate = sum(rmins) / sum(rhours) if sum(rhours) > 0 else float("nan")
                rreps = bootstrap(rhours, rmins, a.n_boot, rng)
                rlo, rhi = (np.percentile(rreps, [2.5, 97.5]) if rreps is not None else (float("nan"), float("nan")))
                rows.append({"body": body, "route": route, "family": fam,
                             "coverage": cov.get(body, "?") if cov.get(body) == "complete" else "partial (floor)",
                             "meetings": len(keys), "meeting_hours": round(sum(hours), 1),
                             "episodes": eps, "episode_minutes": round(sum(mins), 1),
                             "rate_min_per_h": round(rate, 2),
                             "ci95_lo": round(float(lo), 2), "ci95_hi": round(float(hi), 2),
                             "meetings_with_episode": int(sum(1 for x in mins if x > 0)),
                             "meetings_read_by_route": len(rkeys), "hours_read_by_route": round(sum(rhours), 1),
                             "rate_read_denominator": round(rrate, 2),
                             "read_ci95_lo": round(float(rlo), 2), "read_ci95_hi": round(float(rhi), 2)})
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    with open(out / "rates_by_body_route.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    # markdown: complete-coverage bodies only, family = any / density / transit
    md = ["| body | route | meetings | meeting-h | episodes | on-topic min | min per meeting-h (95% boot) | density | transit | read by route: meetings / h | min per read-h (95% boot) |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    lookup = {(r["body"], r["route"], r["family"]): r for r in rows}
    for body in sorted(by_body):
        if cov.get(body) != "complete":
            continue
        for route in sorted({r["route"] for r in rows if r["body"] == body}):
            r = lookup[(body, route, "any")]; d = lookup[(body, route, "density")]; t = lookup[(body, route, "transit")]
            md.append(f"| {body} | {route} | {r['meetings']} | {r['meeting_hours']} | {r['episodes']} | {r['episode_minutes']} "
                      f"| **{r['rate_min_per_h']}** ({r['ci95_lo']}–{r['ci95_hi']}) "
                      f"| {d['rate_min_per_h']} ({d['ci95_lo']}–{d['ci95_hi']}) | {t['rate_min_per_h']} ({t['ci95_lo']}–{t['ci95_hi']}) "
                      f"| {r['meetings_read_by_route']} / {r['hours_read_by_route']} | {r['rate_read_denominator']} ({r['read_ci95_lo']}–{r['read_ci95_hi']}) |")
    (out / "rates_complete_bodies.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    print(f"wrote {out / 'rates_by_body_route.csv'}, {out / 'rates_complete_bodies.md'}")


if __name__ == "__main__":
    main()
