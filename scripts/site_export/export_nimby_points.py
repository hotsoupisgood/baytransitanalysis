#!/usr/bin/env python
"""Export data/nimby_points.json -- one point per speaker cluster, with a quote.

Run from the ytpt repo with its venv so pipeline.state can load .env:

    cd ~/ytpt && .venv/bin/python scripts/site_export/export_nimby_points.py

The scatter and the "Examples" strip in the NIMBY/YIMBY panel read this file.
Nothing here is new scoring: the unit, the frame, the threshold and the role
column are exactly the ones analysis/stance_s1_role_audit_view.py already
computed, and this exporter reads that view's own CSV rather than recomputing
it, so the scatter can never disagree with the bars above it.

  unit     one diarized cluster inside one episode (episode x cluster), with a
           redacted quote; the cluster id carries no person and is never linked
           across meetings (plan/DECISIONS.md #15)
  frame    density-family and both-family episodes; transit episodes are out,
           since YIMBY/NIMBY is a housing construct
  role     role_audit (rule ra-1), the reported role since 2026-10-18. Both
           `public` and `dais` are exported (2026-10-19, for the dot plot);
           staff are inside `dais` because the audit rule puts them there
  label    mean of the cluster's turns' stance_median: YIMBY >= +0.5,
           NIMBY <= -0.5, neutral between; clusters with no numeric turn are
           "no stance" and are not exported
  route    kept on every point and never pooled -- the page shows one route at
           a time, because the routes that read the same body can disagree

The quote is the cluster's longest turn that carries a numeric stance. Its text
comes from `corpus.sqlite` `episode_segments`, rebuilt into turns by the S1
pass's own run-collapsing rule (pipeline/stance.py episode_turns). Two edits are
made to it, in this order: `pipeline/redact.py` replaces personal names (the
self-introduction cues included), street addresses and contact details with
`[name]`, `[address]` and `[contact]` -- owner decision 2026-10-20 09:13, "go
back to removing names", which reverses the 2026-10-18 keep-names rule -- and
then it is trimmed to ~60 words with an ellipsis. Redaction runs first so that a
name near the 60-word boundary cannot survive by being cut in half, and a turn
that was nothing but a name is dropped rather than published as an empty quote.
Turn text is untrusted data: the S1 pass's `contains_instructions` flag travels
with the point so the page can mark it, and nothing in it is ever executed or
followed.

`turn_stance.turn_idx` is the join key, with a (cluster, relative start)
fallback for the ~1 % of rows where a keyword-list change renumbered the index
(the same fallback export_site_data.py uses, and for the same reason).

Never prints .env values or the DB path. Never writes inside the ytpt repo.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from export_site_data import (  # noqa: E402  (shared rules, one definition each)
    CORPUS,
    YTPT,
    clean,
    link_at,
    rows,
    turns_of,
)

SITE = Path(__file__).resolve().parents[2] / "docs"  # the GitHub Pages root of this repo
DATA = SITE / "data"
CLUSTER_CSV = YTPT / "output" / "stance_s1" / "nimby_view" / "speaker_clusters_roleaudit.csv"
OUT = DATA / "nimby_points.json"

#: half a codebook step, the threshold stance_s1_role_audit_view.py labels with
THRESHOLD = 0.5
#: the two role values the view's `role` column carries under rule ra-1. Staff
#: are folded into `dais` by the audit itself, not here.
ROLES = ("public", "dais")
#: the quote is trimmed to this many words, with an ellipsis if it was longer
QUOTE_WORDS = 60
WORD_RE = re.compile(r"[A-Za-z0-9']+")

sys.path.insert(0, str(YTPT))
from pipeline import state  # noqa: E402
from pipeline import redact  # noqa: E402  (the pipeline's own redactor; quotes are published text)
sys.path.insert(0, str(YTPT / "pipeline"))
import stance  # noqa: E402


def trim_words(text, n=QUOTE_WORDS):
    """First n words of a turn, with an ellipsis if anything was dropped."""
    text = re.sub(r"\s{2,}", " ", (text or "").strip())
    parts = text.split(" ")
    if len(parts) <= n:
        return text, False
    return " ".join(parts[:n]).rstrip(",;:.—- ") + " …", True


def hms(s):
    s = int(round(s or 0))
    h, m, x = s // 3600, s % 3600 // 60, s % 60
    return (f"{h}:{m:02d}:{x:02d}" if h else f"{m}:{x:02d}")


def load_clusters():
    """The view's own per-cluster rows: both roles, density/both, with a stance.

    Public-only until 2026-10-19. The dais rows are exported now because the
    dot plot draws the two roles as two halves of one frame, and the point of
    that figure is that neutral is something the dais does -- which is
    invisible if only the public is plotted. Nothing about the public points
    changed: same filter, same fields, same quote rule, same ids. Every point
    now carries `role`, and the stance-by-hostility scatter above the dot plot
    keeps itself to the public half by filtering on it.
    """
    if not CLUSTER_CSV.exists():
        sys.exit(f"missing {CLUSTER_CSV} -- run analysis/stance_s1_nimby_view.py first")
    out = []
    for r in csv.DictReader(CLUSTER_CSV.open(encoding="utf-8")):
        if r["role"] not in ROLES or r["label"] == "no stance" or not r["mean_stance"]:
            continue
        r["mean_stance"] = float(r["mean_stance"])
        r["mean_hostility"] = float(r["mean_hostility"]) if r["mean_hostility"] else None
        for k in ("n_turns", "n_numeric", "words"):
            r[k] = int(float(r[k]))
        out.append(r)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true",
                    help="exit non-zero unless every exported point has a quote and a link")
    a = ap.parse_args()

    con = state.connect()
    cur = con.cursor()
    cp = sqlite3.connect(f"file:{CORPUS}?mode=ro", uri=True)
    cp.row_factory = sqlite3.Row

    pv = stance.prompt_version_for(state.setting("SCORING_MODEL", stance.DEFAULT_MODEL))
    cl = load_clusters()
    eids = sorted({r["episode_id"] for r in cl})
    _byrole = Counter(r["role"] for r in cl)
    print(f"[nimby_points] {len(cl)} clusters with a stance in {len(eids)} episodes "
          f"({', '.join(f'{v} {k}' for k, v in sorted(_byrole.items()))}), prompt_version {pv}")

    # -- episode and meeting context, the same joins the rest of the site uses --
    ep_meta, body_name = {}, {}
    for r in rows(cur, """select e.episode_id, e.start_s, e.method, e.tags, e.agenda_item,
                                 v.speaker as body, v.platform, v.source_url, v.meeting_date, v.title
                          from episodes e join videos v on v.video_id = e.video_id"""):
        ep_meta[r["episode_id"]] = r
    geo_csv = DATA / "bodies_geo.csv"
    if geo_csv.exists():
        for g in csv.DictReader(geo_csv.open(encoding="utf-8")):
            body_name[g["body"]] = g["name"]

    # -- the scored turns of those episodes, keyed both ways --------------------
    by_idx, by_pos = defaultdict(dict), defaultdict(dict)
    q = ("select episode_id, cluster, turn_idx, start_s, end_s, words, stance_median, "
         "hostility_median, frames, sub_topic, contains_instructions "
         "from turn_stance where prompt_version = ? and stance_median is not null")
    for r in rows(cur, q, (pv,)):
        by_idx[r["episode_id"]][r["turn_idx"]] = r
        by_pos[r["episode_id"]][(r["cluster"], round(r["start_s"] or 0, 1))] = r

    # -- the corpus turns, rebuilt by the S1 pass's own collapsing rule ---------
    segs = defaultdict(list)
    qs = ("select episode_id, seg_idx, start_s, end_s, speaker, text from episode_segments "
          f"where episode_id in ({','.join('?' * len(eids))}) order by episode_id, seg_idx")
    for s in cp.execute(qs, eids):
        segs[s["episode_id"]].append(s)
    corpus_turns = {}
    for eid, ss in segs.items():
        corpus_turns[eid] = turns_of([dict(x) for x in ss])

    stats = defaultdict(int)
    points = []
    pending = []
    for c in cl:
        eid, clus = c["episode_id"], c["cluster"]
        meta = ep_meta.get(eid)
        if not meta:
            stats["no_episode_row"] += 1
            continue
        base = meta["start_s"] or 0          # corpus segment clocks are episode-relative

        # the cluster's scored turns, matched to their corpus text
        cand = []
        for t in corpus_turns.get(eid, []):
            if t["cluster"] != clus:
                continue
            sr = by_idx[eid].get(t["turn_idx"])
            if not (sr and sr["cluster"] == clus and abs((sr["start_s"] or 0) - t["start_rel"]) <= 1.5):
                sr = by_pos[eid].get((clus, t["start_rel"]))
                if sr is not None:
                    stats["matched_by_position"] += 1
            if sr and sr["cluster"] == clus:
                cand.append((t, sr))
        if not cand:
            stats["no_quotable_turn"] += 1
            continue
        t, sr = max(cand, key=lambda ts: (ts[0]["words"], -ts[0]["start_rel"]))
        at = round(base + t["start_rel"], 1)
        link = link_at(meta["platform"], meta["source_url"], at)
        if not link:
            stats["no_link"] += 1
            continue

        pending.append({
            "_text": t["text"],
            "id": f"{eid}/{clus}",
            "episode_id": eid,
            "cluster": clus,
            "body": meta["body"],
            "body_name": body_name.get(meta["body"], meta["body"]),
            "role": c["role"],
            "route": c["method"],
            "family": c["family"],
            "date": meta["meeting_date"],
            "label": c["label"],
            "mean_stance": round(c["mean_stance"], 3),
            "mean_hostility": None if c["mean_hostility"] is None else round(c["mean_hostility"], 3),
            "n_turns": c["n_turns"],
            "n_scored": c["n_numeric"],
            "words": c["words"],
            "quote_words": t["words"],
            "quote_stance": sr["stance_median"],
            "quote_hostility": sr["hostility_median"],
            "sub_topic": sr["sub_topic"] or "",
            "frames": [f for f in (sr["frames"] or "").split(";") if f],
            "flagged": bool(sr["contains_instructions"]),
            "at_s": at,
            "at_hms": hms(at),
            "link": link,
            "platform": meta["platform"],
        })

    # -- redact, then trim ------------------------------------------------------
    # Owner, 2026-10-20 09:13: names out of everything published. These quotes are the
    # most exposed text on the site -- a self-introduction is exactly the sentence a
    # speaker opens with, and the longest-turn rule tends to pick the turn that has one.
    # The spaCy NER pass is the slow part, so the chosen turns are batched through
    # nlp.pipe() once here rather than one call per cluster inside the loop above.
    model = redact.nlp()
    if model is None:
        sys.exit("pipeline/redact.py could not load spaCy en_core_web_sm -- refusing to "
                 "publish quotes the person-name layer never saw")
    texts = [p["_text"] for p in pending]
    docs = model.pipe(texts, batch_size=64)
    redaction = Counter()
    for p, text, sdoc in zip(pending, texts, docs):
        # publish=True, 2026-10-20: at the default FULL tier the name layers
        # protect a lone token that is also a Bay Area place, so first names that
        # double as place names reached the published quotes in clear (Carlos,
        # Howard, Linda, Susan, George, Benjamin, Anthony -- 21 quotes). The
        # publish tiers keep the multi-word place protected, so "San Carlos" stays.
        red, reps = redact.redact_text(text, doc=sdoc, publish=True)
        redaction.update(r["rule"] for r in reps)
        quote, trimmed = trim_words(red)
        # Second pass, on the string that actually gets published. The NER layer is
        # context-sensitive, so a name the model let through while reading the whole
        # turn can become the obvious PERSON of a 60-word excerpt -- 13 of these
        # survived the first pass alone. Re-running on the trimmed quote is cheap
        # (short strings), idempotent (PLACEHOLDER_RE skips the `[name]` already
        # inserted), and it is the last thing to touch the text before it is written.
        if quote:
            quote, reps2 = redact.redact_text(quote, publish=True)
            redaction.update(f"{r['rule']}_on_trim" for r in reps2)
        if not quote:
            # the whole turn was a name and nothing else
            stats["empty_quote_after_redaction"] += 1
            continue
        p.pop("_text")
        p["quote"] = quote
        p["quote_trimmed"] = trimmed
        points.append(p)
    print(f"[redact] {redact.VERSION}: {sum(redaction.values())} replacements in "
          f"{len(pending)} quotes; by rule {dict(sorted(redaction.items())) or 'nothing'}")

    # `routes` stays public-only and keeps its exact shape: the stance-by-hostility
    # scatter reads it for the route picker's "n speakers" labels, and that chart is
    # still about public comment. The dot plot, which draws both halves, reads
    # `routes_by_role` instead -- so adding the dais could not silently restate a
    # number the public scatter was already showing.
    by_route = defaultdict(lambda: defaultdict(int))
    by_route_role = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    for p in points:
        if p["role"] == "public":
            by_route[p["route"]][p["label"]] += 1
            by_route[p["route"]]["n"] += 1
        d = by_route_role[p["route"]][p["role"]]
        d[p["label"]] += 1
        d["n"] += 1
    doc = {
        "meta": {
            # What this file actually is, now that the quotes are redacted: the cluster id
            # is per episode and carries no person, and the quote has had names, addresses
            # and contact details removed -- so a point is anonymous, and saying so is not
            # a claim about the raw corpus. DECISIONS #15 is cited for the one thing it
            # governs: clusters are never linked across meetings.
            "unit": "one diarized speaker cluster inside one episode, with its quote "
                    "redacted; the cluster id is per episode and is never linked across "
                    "meetings (plan/DECISIONS.md #15)",
            "redaction": {"version": redact.VERSION,
                          "by_rule": dict(sorted(redaction.items())),
                          "total": sum(redaction.values()),
                          "rule": "names (self-introduction cues included), street addresses "
                                  "and contact details removed before the 60-word trim "
                                  "(owner, 2026-10-20 09:13)"},
            "role_column": "role_audit (rule ra-1)",
            "roles": list(ROLES),
            "role_note": ("Both roles are exported. Staff speakers are folded into "
                          "`dais` by the audit rule itself, not here."),
            "frame": "density-family and both-family episodes only",
            "threshold": THRESHOLD,
            "prompt_version": pv,
            "quote_rule": f"the cluster's longest turn with a numeric stance, trimmed to {QUOTE_WORDS} words",
            "quote_words_max": QUOTE_WORDS,
            "stamp": "EXPLORATORY, stance κ gate not passed",
            "source": "output/stance_s1/nimby_view/speaker_clusters_roleaudit.csv",
            "n_clusters_in_view": len(cl),
            "n_points": len(points),
            "n_points_public": sum(1 for p in points if p["role"] == "public"),
            "n_points_dais": sum(1 for p in points if p["role"] == "dais"),
            "dropped": dict(stats),
            "routes": {r: dict(v) for r, v in sorted(by_route.items())},
            "routes_by_role": {r: {ro: dict(v) for ro, v in sorted(d.items())}
                               for r, d in sorted(by_route_role.items())},
            "bodies": sorted({p["body"] for p in points}),
        },
        "points": sorted(points, key=lambda p: (p["route"], p["role"], p["body"],
                                                p["date"] or "", p["id"])),
    }
    OUT.write_text(json.dumps(clean(doc), indent=0, ensure_ascii=False), encoding="utf-8")
    print(f"[nimby_points] wrote {OUT.relative_to(SITE)}: {len(points)} points, "
          f"{OUT.stat().st_size/1024:.0f} kB; dropped {dict(stats) or 'nothing'}")
    for r, d in sorted(by_route_role.items()):
        for ro, v in sorted(d.items()):
            print(f"   {r:<18} {ro:<7} {v['n']:>4} points  YIMBY {v.get('YIMBY',0)}"
                  f"  NIMBY {v.get('NIMBY',0)}  neutral {v.get('neutral',0)}")

    con.close()
    cp.close()

    if a.check:
        bad = [p["id"] for p in points if not (p["quote"] and p["link"] and p["at_hms"])]
        miss = len(cl) - len(points)
        ok = not bad and points and miss <= 0.02 * len(cl)
        print(f"[check] {len(points)} points, {len(bad)} without a quote or link, "
              f"{miss} clusters dropped of {len(cl)}")
        if bad:
            print("  first few: " + ", ".join(bad[:5]))
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
