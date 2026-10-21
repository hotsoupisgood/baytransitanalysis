#!/usr/bin/env python3
"""
scripts/stance_s2_sample.py -- the S2 top-up frame for the stance pass
(plan/STANCE-S2.md). S1 (plan/STANCE-S1-RESULTS.md 2-4) scored the whole frame
as it stood on 2026-10-18: 9,358 turns over 628 QA-gated corpus episodes. The
corpus has since been reloaded from output/whisper (1,209 episodes), so the
frame has grown. S2 is a **census of the remainder** -- there is nothing to
stratify, every unscored frame turn is sent -- under the same frame rule:

  * episode in corpus.sqlite (which is built from the QA gate output/qa/latest.csv)
  * family density or both (transit episodes are a different estimand;
    YIMBY/NIMBY is a housing construct, S1 7b / NIMBY-PILOT 5)
  * a redacted transcript exists in output/redacted/
  * turn unit = pipeline/stance.py episode_turns, turns >= 5 words
  * role (cue role, corpus `episode_speakers`) dais or public; `unknown` is
    outside the estimand exactly as in S1
  * no turn_stance row yet under the pinned prompt_version

The per-body 8 h cap of S1 7b/10 is a *corpus-construction* rule (it decided
which meetings were transcribed, upstream of the stance pass); it is not
re-applied here, because S2 adds no new transcription -- it scores transcripts
the cap already admitted. Stated in plan/STANCE-S2.md so the frame rule stays
auditable.

Positional-id safety (memory: keyword-list changes renumber episode ids and
turn indices). A turn is treated as already scored only when the existing
turn_stance row agrees on cluster at that turn_idx. An episode where any
scored turn_idx now carries a different cluster is a *re-cut*: it is reported
and excluded wholesale, so no existing row is shadowed by a renumbered turn
and no row is ever modified.

    scripts/stance_s2_sample.py --pool-only       # counts + estimate, writes nothing
    scripts/stance_s2_sample.py --out output/stance_s2
      writes s2_frame.csv   (every S2-frame turn, `scored` flag -- the join
                             table for scripts/role_audit.py and the views)
             s2_new_turns.csv (the unscored ones, in submit order)
             frame_all.csv  (s1_frame.csv + s2_frame.csv, deduped: the frame
                             CSV the views join against after S2)
             s2_strata.csv, s2_manifest.json
"""
import argparse, json, sqlite3, sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))
import stance, state  # noqa: E402

CORPUS = Path("~/.local/share/ytpt/corpus.sqlite").expanduser()
REDACTED = ROOT / "output" / "redacted"
S1_FRAME = ROOT / "output" / "stance_s1" / "s1_frame.csv"
FAMILIES = ("density", "both")
ROLES = ("dais", "public")
MEASURED_USD_PER_CALL = 0.001305          # S1 actual, Batches, 98 % cache hits (S1 3)
COLS = stance.SAMPLE_COLS + ["method", "roles_source", "stratum", "scored"]


def corpus_turns(corpus=CORPUS, redacted=REDACTED, min_words=5, families=FAMILIES):
    con = sqlite3.connect(f"file:{corpus}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    roles = defaultdict(dict)
    for r in con.execute("SELECT episode_id, speaker, role FROM episode_speakers"):
        roles[r["episode_id"]][r["speaker"]] = r["role"]
    eps = [dict(r) for r in con.execute("SELECT episode_id, body, family, method, roles_source, agenda_item "
                                        "FROM episodes ORDER BY episode_id")]
    con.close()
    n_family = 0
    out, missing = [], []
    for e in eps:
        if e["family"] not in families:
            continue
        n_family += 1
        p = redacted / f"{e['episode_id']}.json"
        if not p.exists():
            missing.append(e["episode_id"])
            continue
        doc = json.loads(p.read_text(encoding="utf-8"))
        meta = {"body": e["body"], "family": e["family"], "agenda_item": e["agenda_item"]}
        for t in stance.episode_turns(e["episode_id"], doc, roles.get(e["episode_id"], {}), meta):
            t["method"] = e["method"] or "unknown"
            t["roles_source"] = e["roles_source"] or "unknown"
            t["stratum"] = f"{t['body']}|{t['role']}|{t['method']}"
            if t["words"] >= min_words:
                out.append(t)
    return out, missing, len(eps), n_family


def scored_keys(pv):
    con = state.connect(None, create=False)
    rows = [dict(r) for r in con.execute("SELECT episode_id, cluster, turn_idx FROM turn_stance WHERE prompt_version = ?", (pv,))]
    con.close()
    return {f"{r['episode_id']}__t{r['turn_idx']}": r for r in rows}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--min-words", type=int, default=5)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--model", default=None)
    ap.add_argument("--out", default=str(ROOT / "output" / "stance_s2"))
    ap.add_argument("--pool-only", action="store_true")
    a = ap.parse_args(argv)
    model = a.model or state.setting("SCORING_MODEL", stance.DEFAULT_MODEL)
    pv = stance.prompt_version_for(model)

    pool, missing, n_eps, n_family = corpus_turns(min_words=a.min_words)
    pool = [t for t in pool if t["role"] in ROLES]
    scored = scored_keys(pv)

    # positional-id safety: find episodes whose scored turn_idx now names another cluster
    by_id = {t["turn_id"]: t for t in pool}
    recut = set()
    matched = 0
    for key, r in scored.items():
        t = by_id.get(key)
        if t is None:
            continue
        if t["cluster"] != r["cluster"]:
            recut.add(r["episode_id"])
        else:
            matched += 1
    pool = [t for t in pool if t["episode_id"] not in recut]
    for t in pool:
        k = t["turn_id"]
        t["scored"] = int(k in scored and scored[k]["cluster"] == t["cluster"])
    new = sorted((t for t in pool if not t["scored"]), key=lambda t: t["turn_id"])

    ep_pool = {t["episode_id"] for t in pool}
    ep_scored = {t["episode_id"] for t in pool if t["scored"]}
    print(f"corpus: {n_eps} episodes, {n_family} density/both, {len(missing)} without a redacted file")
    print(f"S2 frame: {len(pool)} turns >= {a.min_words} words, role in {ROLES}, over {len(ep_pool)} episodes; "
          + ", ".join(f"{r} {n}" for r, n in sorted(Counter(t['role'] for t in pool).items())))
    print(f"already scored under {pv}: {len(scored)} turn_stance rows, {matched} matched in this frame by key+cluster; "
          f"{len(ep_scored)} frame episodes carry scored turns")
    print(f"re-cut episodes excluded (scored turn_idx now a different cluster): {len(recut)}"
          + (" " + ", ".join(sorted(recut)[:10]) if recut else ""))
    print(f"S2 new turns: {len(new)}; " + ", ".join(f"{r} {n}" for r, n in sorted(Counter(t['role'] for t in new).items())))
    print("new by body: " + ", ".join(f"{b} {n}" for b, n in Counter(t["body"] for t in new).most_common()))
    print("new by route: " + ", ".join(f"{m} {n}" for m, n in Counter(t["method"] for t in new).most_common()))
    print("new by route x role: " + ", ".join(f"{k[0]}/{k[1]} {n}" for k, n in sorted(Counter((t['method'], t['role']) for t in new).items())))

    words = sum(len(stance.user_turn(t).split()) for t in new)
    est = stance.estimate_usd(a.k * words, a.k * len(new), model, batch=True)
    est_measured = MEASURED_USD_PER_CALL * a.k * len(new)
    print(f"estimate for {len(new)} turns x k={a.k} = {a.k * len(new)} batch calls ({words:,} prompt words): "
          f"stance.estimate_usd ${est:.2f}; at S1's measured ${MEASURED_USD_PER_CALL:.6f}/call ${est_measured:.2f}")
    if a.pool_only:
        return 0

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    stance.write_csv(out / "s2_frame.csv", pool, COLS)
    stance.write_csv(out / "s2_new_turns.csv", new, COLS)
    # the frame CSV the views join against after S2: S1's frame plus this one, deduped
    import csv as _csv
    seen, rows_all = set(), []
    for src in (S1_FRAME, out / "s2_frame.csv"):
        with open(src, newline="", encoding="utf-8") as f:
            for r in _csv.DictReader(f):
                key = (r["episode_id"], r["cluster"], r["turn_idx"])
                if key in seen:
                    continue
                seen.add(key)
                rows_all.append({c: r.get(c, "") for c in COLS})
    stance.write_csv(out / "frame_all.csv", rows_all, COLS)
    strata = defaultdict(lambda: {"frame": 0, "scored": 0, "new": 0})
    for t in pool:
        strata[t["stratum"]]["frame"] += 1
        strata[t["stratum"]]["scored" if t["scored"] else "new"] += 1
    rows = [{"stratum": k, "body": k.split("|")[0], "role": k.split("|")[1], "route": k.split("|")[2], **v}
            for k, v in sorted(strata.items())]
    stance.write_csv(out / "s2_strata.csv", rows, ["stratum", "body", "role", "route", "frame", "scored", "new"])
    manifest = {"min_words": a.min_words, "k": a.k, "model": model, "prompt_version": pv, "corpus": str(CORPUS),
                "redacted_dir": str(REDACTED), "families": list(FAMILIES), "roles": list(ROLES),
                "corpus_episodes": n_eps, "corpus_episodes_family": n_family,
                "episodes_without_redacted_file": missing, "recut_episodes_excluded": sorted(recut),
                "frame_turns": len(pool), "frame_episodes": len(ep_pool),
                "frame_by_role": dict(Counter(t["role"] for t in pool)),
                "already_scored_rows": len(scored), "already_scored_matched": matched,
                "new_turns": len(new), "new_by_role": dict(Counter(t["role"] for t in new)),
                "new_by_body": dict(Counter(t["body"] for t in new).most_common()),
                "new_by_route": dict(Counter(t["method"] for t in new).most_common()),
                "prompt_words_new": words, "estimate_usd_stance_py": round(est, 2),
                "estimate_usd_measured_rate": round(est_measured, 2),
                "frame_all_rows": len(rows_all)}
    (out / "s2_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n")
    print(f"wrote {out}/s2_frame.csv, s2_new_turns.csv, frame_all.csv ({len(rows_all)} rows), s2_strata.csv, s2_manifest.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
