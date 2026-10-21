#!/usr/bin/env python3
"""
scripts/stance_s1_sample.py -- the stratified S1 sample for the stance pass
(plan/STANCE-S1-RESULTS.md). Draws turns from the QA-gated, REDACTED corpus
(~/.local/share/ytpt/corpus.sqlite + output/redacted/<episode>.json), with the
pilot's turn unit (pipeline/stance.py episode_turns: a maximal run of
consecutive segments by one cluster inside one episode, turns >= 5 words),
roles from the corpus `episode_speakers` table (roles_source carried as a
facet), locate route (`episodes.method`) carried as the stratum.

Strata: body x role x route. Allocation: every public turn is taken (public
is the scarce role; this satisfies ">= 100 public turns where a body has
them"); the dais budget is n_total - n_public, allocated over bodies with a
floor of --floor-dais per body (or everything a smaller body has) and the
remainder proportional to body size, then within a body proportional over
routes. The 300 pilot turns (turn_stance rows under the pinned
prompt_version) are forced into their cells and never re-scored.

    scripts/stance_s1_sample.py --n 10000 --out output/stance_s1
    writes s1_turns.csv (all sampled turns, pilot flagged), s1_new_turns.csv
    (the ones to send), s1_strata.csv, s1_manifest.json
"""
import argparse, json, random, sqlite3, sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))
import stance, state  # noqa: E402

CORPUS = Path("~/.local/share/ytpt/corpus.sqlite").expanduser()
REDACTED = ROOT / "output" / "redacted"
SEED = 20261018
ROLES = ("dais", "public")
PILOT_USD_PER_CALL = 1.9805 / 890          # the pilot's Message Batch, measured 2026-10-17
COLS = stance.SAMPLE_COLS + ["method", "roles_source", "stratum", "pilot"]


def corpus_turns(corpus=CORPUS, redacted=REDACTED, min_words=5):
    con = sqlite3.connect(f"file:{corpus}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    roles = defaultdict(dict)
    for r in con.execute("SELECT episode_id, speaker, role FROM episode_speakers"):
        roles[r["episode_id"]][r["speaker"]] = r["role"]
    eps = [dict(r) for r in con.execute("SELECT episode_id, body, family, method, roles_source, agenda_item, transcript_path "
                                        "FROM episodes ORDER BY episode_id")]
    con.close()
    out, missing, paths = [], [], Counter()
    for e in eps:
        paths[str(Path(e["transcript_path"] or "").parent)] += 1
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
    return out, missing, paths, len(eps)


def pilot_keys(pv):
    con = state.connect(None, create=False)
    # the pilot rows only: aggregated 2026-10-17; S1 rows are written on/after 2026-10-18 and must not count as pilot
    rows = [dict(r) for r in con.execute("SELECT episode_id, cluster, turn_idx, role, body, words FROM turn_stance "
                                         "WHERE prompt_version = ? AND updated_at < '2026-10-18'", (pv,))]
    con.close()
    return {f"{r['episode_id']}__t{r['turn_idx']}": r for r in rows}


def allocate_s1(pool, pilot, n_total, floor_dais, seed):
    rng = random.Random(seed)
    by_cell = defaultdict(list)
    for t in pool:
        by_cell[(t["body"], t["role"], t["method"])].append(t)
    chosen = []
    # public: everything
    n_public = 0
    for key, rows in by_cell.items():
        if key[1] == "public":
            chosen += rows
            n_public += len(rows)
    # dais: body floor + proportional, then proportional over routes inside the body
    dais_budget = max(0, n_total - n_public)
    body_sizes = Counter()
    for key, rows in by_cell.items():
        if key[1] == "dais":
            body_sizes[key[0]] += len(rows)
    body_alloc = stance.allocate(dict(body_sizes), dais_budget, floor=floor_dais)
    alloc = {}
    for body, nb in body_alloc.items():
        route_sizes = {key[2]: len(rows) for key, rows in by_cell.items() if key[0] == body and key[1] == "dais"}
        for route, nr in stance.allocate(route_sizes, nb, floor=1).items():
            alloc[(body, "dais", route)] = nr
    for key, nr in alloc.items():
        rows = sorted(by_cell[key], key=lambda t: t["turn_id"])
        forced = [t for t in rows if t["turn_id"] in pilot]
        rest = [t for t in rows if t["turn_id"] not in pilot]
        rng.shuffle(rest)
        chosen += forced + rest[:max(0, nr - len(forced))]
    for t in chosen:
        t["pilot"] = int(t["turn_id"] in pilot)
    chosen.sort(key=lambda t: t["turn_id"])
    return chosen, {"public": n_public, "dais_budget": dais_budget, "body_alloc": body_alloc}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--floor-dais", type=int, default=150)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--min-words", type=int, default=5)
    ap.add_argument("--model", default=None)
    ap.add_argument("--out", default=str(ROOT / "output" / "stance_s1"))
    ap.add_argument("--pool-only", action="store_true", help="print pool sizes and estimates, write nothing")
    a = ap.parse_args(argv)
    model = a.model or state.setting("SCORING_MODEL", stance.DEFAULT_MODEL)
    pv = stance.prompt_version_for(model)
    pool, missing, paths, n_eps = corpus_turns(min_words=a.min_words)
    pool = [t for t in pool if t["role"] in ROLES]
    pilot = pilot_keys(pv)
    in_pool = {t["turn_id"]: t for t in pool}
    matched = {k: v for k, v in pilot.items() if k in in_pool and in_pool[k]["cluster"] == v["cluster"]}
    role_diff = [k for k, v in matched.items() if in_pool[k]["role"] != v["role"]]
    print(f"corpus: {n_eps} episodes, transcript dirs {dict(paths)}, {len(missing)} without a redacted file")
    print(f"pool: {len(pool)} turns >= {a.min_words} words with role in {ROLES}; "
          + ", ".join(f"{r} {n}" for r, n in sorted(Counter(t['role'] for t in pool).items())))
    print(f"pilot (prompt_version {pv}): {len(pilot)} turn_stance rows, {len(matched)} found in the pool by key+cluster, "
          f"{len(role_diff)} with a different role now, {len(pilot) - len(matched)} not in the pool")
    pub_by_body = Counter(t["body"] for t in pool if t["role"] == "public")
    print("public turns by body: " + ", ".join(f"{b} {n}" for b, n in pub_by_body.most_common()))
    print("dais turns by body: " + ", ".join(f"{b} {n}" for b, n in Counter(t['body'] for t in pool if t['role'] == 'dais').most_common()))
    print("route x role: " + ", ".join(f"{k[0]}/{k[1]} {n}" for k, n in sorted(Counter((t['method'], t['role']) for t in pool).items())))
    chosen, info = allocate_s1(pool, matched, a.n, a.floor_dais, a.seed)
    new = [t for t in chosen if not t["pilot"]]
    words = sum(len(stance.user_turn(t).split()) for t in new)      # per turn; every turn is sent k=3 times
    est = stance.estimate_usd(3 * words, 3 * len(new), model, batch=True)
    est_pilot_rate = PILOT_USD_PER_CALL * 3 * len(new)
    print(f"sample: {len(chosen)} turns ({sum(t['pilot'] for t in chosen)} pilot reused, {len(new)} new); "
          + ", ".join(f"{r} {n}" for r, n in sorted(Counter(t['role'] for t in chosen).items())))
    print(f"estimate for {len(new)} new turns x k=3 = {3 * len(new)} batch calls ({words:,} prompt words): "
          f"stance.estimate_usd ${est:.2f}; at the pilot's measured ${PILOT_USD_PER_CALL:.5f}/call ${est_pilot_rate:.2f}")
    if a.pool_only:
        return 0
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    stance.write_csv(out / "s1_turns.csv", chosen, COLS)
    stance.write_csv(out / "s1_new_turns.csv", new, COLS)
    # the whole frame (census) and its unsampled remainder, for a top-up under the same ceiling
    chosen_ids = {t["turn_id"] for t in chosen}
    for t in pool:
        t["pilot"] = int(t["turn_id"] in matched)
        t["sampled"] = int(t["turn_id"] in chosen_ids)
    stance.write_csv(out / "s1_frame.csv", pool, COLS + ["sampled"])
    stance.write_csv(out / "s1_rest_turns.csv", [t for t in pool if not t["sampled"]], COLS + ["sampled"])
    strata = defaultdict(lambda: {"pool": 0, "sampled": 0, "pilot": 0})
    for t in pool:
        strata[t["stratum"]]["pool"] += 1
    for t in chosen:
        strata[t["stratum"]]["sampled"] += 1
        strata[t["stratum"]]["pilot"] += t["pilot"]
    rows = [{"stratum": k, "body": k.split("|")[0], "role": k.split("|")[1], "route": k.split("|")[2], **v} for k, v in sorted(strata.items())]
    stance.write_csv(out / "s1_strata.csv", rows, ["stratum", "body", "role", "route", "pool", "sampled", "pilot"])
    manifest = {"seed": a.seed, "n_requested": a.n, "floor_dais": a.floor_dais, "min_words": a.min_words, "model": model,
                "prompt_version": pv, "corpus": str(CORPUS), "redacted_dir": str(REDACTED), "corpus_episodes": n_eps,
                "episodes_without_redacted_file": missing, "pool_turns": len(pool),
                "pool_by_role": dict(Counter(t["role"] for t in pool)), "pilot_rows": len(pilot), "pilot_matched": len(matched),
                "pilot_role_changed": role_diff, "sampled": len(chosen), "sampled_new": len(new),
                "sampled_by_role": dict(Counter(t["role"] for t in chosen)), "allocation": info,
                "prompt_words_new": words, "estimate_usd_stance_py": round(est, 2), "estimate_usd_pilot_rate": round(est_pilot_rate, 2)}
    (out / "s1_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n")
    print(f"wrote {out}/s1_turns.csv, s1_new_turns.csv, s1_strata.csv, s1_manifest.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
