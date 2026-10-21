#!/usr/bin/env python3
"""
scripts/codebook_v2_pilot.py -- pilot of codebook v2 (classify.INSTRUCTIONS_V2)
and of 120-s candidate windows against the exhaustive recall labels
(plan/CODEBOOK-V2-PILOT.md, 2026-10-18; approved by Puck, $6 ceiling).

The reference is output/validation/recall/post_kw2/windows.jsonl: every 60-s
window of the four usable whole-meeting GPU transcripts, labelled with the
PINNED codebook (t0-a4a8e55760). It is not re-labelled. Only the pipeline side
is re-run, per ARM, on the Mac-ASR locate track exactly as the pipeline reads
it (expanded keyword screen kw-24e8a4acd1, 0.5 hits/100w, pad 30 s):

    v1_240   pinned codebook, 240-s windows  -- the pipeline as it is; its rows
             are the real classifier_calls (video_id = <vid>), nothing is billed
    v1_120   pinned codebook, 120-s windows  -- edit 3 alone
    v2_120   codebook v2,     120-s windows  -- edits 1-3
    v2_240   codebook v2,     240-s windows  -- edits 1-2 alone (optional)

Pilot rows go to classifier_calls under video_id 'cbv2:<vid>' and the arm's own
prompt_version (classify.prompt_version_for(..., codebook, max_run)), so no
per-meeting $ SELECT sees them and nothing pools with the instrument; the
`episodes` table is never written -- merged spans live in this directory.

    estimate                      candidates per arm, $ estimate, nothing sent
    submit  --arms v1_120,v2_120  one Message Batch per arm (status codebook_v2_pilot)
    collect                       labels -> classifier_calls + windows_<arm>.jsonl
    analyze                       recall / precision per family per arm, over-calls,
                                  recovered and remaining misses, results.json, *.md
"""
import argparse, difflib, json, sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))
sys.path.insert(0, str(ROOT / "scripts"))
import state            # noqa: E402
import segment as sg    # noqa: E402
import classify as cl   # noqa: E402
import recall_check as rc   # noqa: E402

OUT = ROOT / "output" / "validation" / "codebook_v2"
REF = ROOT / "output" / "validation" / "recall" / "post_kw2"
ORIG = ROOT / "output" / "validation" / "recall"          # the pre-expansion run: the 30 original misses
CALL_PREFIX = "cbv2:"
API_PREFIX = "cbv2-"
BATCH_STATUS = "codebook_v2_pilot"
NOTE = "pilot-v2: codebook v2 pilot (plan/CODEBOOK-V2-PILOT.md)"
ARMS = {"v1_240": ("v1", 240.0), "v1_120": ("v1", 120.0), "v2_120": ("v2", 120.0), "v2_240": ("v2", 240.0)}
W, PAD, GAP, HALF = 60.0, 30.0, 90.0, 30.0
CAPTIONS_KIND = "asr_local"


def log(msg):
    print(f"{state.now_iso()} {msg}", flush=True)


def model_id():
    return state.setting("SCORING_MODEL", cl.LEGACY_MODEL)


def pv_for(arm, model=None):
    cb, mr = ARMS[arm]
    return cl.prompt_version_for(model or model_id(), cl.DEFAULT_LANG, CAPTIONS_KIND, cb, mr)


def params_for(arm):
    cb, mr = ARMS[arm]
    return dict(cl.DEFAULTS, max_run=mr, pad=PAD, codebook=cb)


def load_ref():
    rows = [json.loads(l) for l in (REF / "windows.jsonl").read_text().splitlines() if l.strip()]
    sample = json.loads((REF / "sample.json").read_text())
    usable = [m for m in sample["meetings"] if m.get("transcript") and m.get("ingest_status") == "done"]
    return sample, usable, [r for r in rows if r.get("label")]


def arm_candidates(con, arm, meetings, keywords):
    """{vid: (cands, {id: row-or-None})} for one arm on the locate track."""
    model = model_id()
    p = params_for(arm)
    out = {}
    for m in meetings:
        vid = m["video_id"]
        segs = sg.read_segments(m["vtt"])
        cands = cl.candidates(vid, segs, keywords, p)
        if arm == "v1_240":
            have = cl.cached_calls(con, vid, model, pv_for(arm, model), cands)      # the pipeline's own rows
        else:
            for c in cands:
                c["id"] = CALL_PREFIX + c["id"]
            have = cl.cached_calls(con, CALL_PREFIX + vid, model, pv_for(arm, model), cands)
        out[vid] = (cands, {c["id"]: have.get(c["id"]) for c in cands})
    return out


def pilot_path():
    return OUT / "pilot.json"


def load_pilot():
    return json.loads(pilot_path().read_text()) if pilot_path().exists() else {"arms": {}}


def save_pilot(d):
    OUT.mkdir(parents=True, exist_ok=True)
    pilot_path().write_text(json.dumps(d, indent=1))


# --- estimate / submit / collect ----------------------------------------------------

def estimate(con, arms, verbose=True):
    sample, usable, _ = load_ref()
    keywords = cl.keywords_for("en")
    model = model_id()
    pilot = load_pilot()
    pilot.update({"model": model, "keywords_version": keywords.version, "reference": str(REF),
                  "meetings": [m["video_id"] for m in usable],
                  "codebook_versions": dict(cl.CODEBOOK_VERSIONS)})
    total_est = total_worst = 0.0
    for arm in arms:
        cb, mr = ARMS[arm]
        per = arm_candidates(con, arm, usable, keywords)
        todo = [c for vid, (cands, have) in per.items() for c in cands if have[c["id"]] is None]
        n_all = sum(len(c) for c, _ in per.values())
        tokens = sum(cl.caption_tokens_for(len(c["text"].split())) for c in todo)
        est = cl.estimate_usd(tokens, len(todo), model, batch=True, codebook=cb)
        worst = sum(cl.worst_case_usd(c["text"], model, batch=True, codebook=cb) for c in todo)
        a = pilot["arms"].setdefault(arm, {})
        a.update({"codebook": cb, "max_run_s": mr, "prompt_version": pv_for(arm, model), "n_candidates": n_all,
                  "n_to_label": len(todo), "padded_words": sum(len(c["text"].split()) for c in todo),
                  "est_usd": round(est, 4), "worst_usd": round(worst, 4)})
        total_est += est
        total_worst += worst
        if verbose:
            log(f"{arm}: codebook {cb} ({cl.CODEBOOK_VERSIONS[cb]}), max-run {mr:.0f} s, prompt_version {a['prompt_version']}: "
                f"{n_all} candidate windows on {len(per)} meetings, {len(todo)} unlabelled ({a['padded_words']:,} padded words, "
                f"~{tokens:,} tokens); estimate ${est:.3f}, worst ${worst:.3f} at the batch rate")
    pilot["estimate_total_usd"] = round(total_est, 4)
    pilot["worst_total_usd"] = round(total_worst, 4)
    save_pilot(pilot)
    if verbose:
        log(f"TOTAL estimate ${total_est:.3f} (worst ${total_worst:.3f}) for arms {', '.join(arms)}")
    return pilot


def submit(con, arms, cap_usd, dry_run=False):
    import spend                                               # noqa: E402
    pilot = estimate(con, arms)
    if pilot["estimate_total_usd"] > cap_usd:
        log(f"REFUSED: estimate ${pilot['estimate_total_usd']:.2f} exceeds the ${cap_usd} ceiling")
        return 2
    why = spend.claude_refusal(con, pilot["worst_total_usd"])
    if why:
        log(f"REFUSED: {why}")
        return 2
    if dry_run:
        log("dry run: nothing submitted")
        return 0
    sample, usable, _ = load_ref()
    keywords = cl.keywords_for("en")
    model = model_id()
    import anthropic
    client = anthropic.Anthropic()
    for arm in arms:
        if arm == "v1_240":
            continue                                           # the pipeline's own rows; never re-billed here
        cb, mr = ARMS[arm]
        pv = pv_for(arm, model)
        per = arm_candidates(con, arm, usable, keywords)
        now = state.now_iso()
        provisional = f"pending-{API_PREFIX}{arm}-{now}"
        reqs, recs = [], []
        for vid, (cands, have) in per.items():
            for c in cands:
                if have[c["id"]] is not None:
                    continue
                recs.append(cl.call_record(c, CALL_PREFIX + vid, model, pv, batch_id=provisional))
                reqs.append({"custom_id": API_PREFIX + arm + "-" + c["id"][len(CALL_PREFIX):],
                             "params": cl.request_params(model, c["text"], cl.DEFAULT_LANG, cb)})
        if not reqs:
            log(f"{arm}: nothing to submit")
            continue
        con.execute("INSERT INTO classifier_batches (batch_id, created_at, status, model, prompt_version, n_requests, est_usd, note) "
                    "VALUES (?,?,?,?,?,?,?,?)", (provisional, now, BATCH_STATUS, model, pv, len(reqs),
                                                 pilot["arms"][arm]["est_usd"], f"{NOTE}; arm {arm}; codebook {cb}; max_run {mr:.0f}"))
        for rec in recs:
            cl.insert_call(con, cl.call_row(rec), replace=True)
        try:
            mb = client.messages.batches.create(requests=reqs)
        except Exception as e:                                 # noqa: BLE001 -- nothing billed; undo
            con.execute("DELETE FROM classifier_calls WHERE batch_id = ? AND label IS NULL", (provisional,))
            con.execute("DELETE FROM classifier_batches WHERE batch_id = ?", (provisional,))
            log(f"{arm}: batch create failed: {type(e).__name__}: {str(e)[:200]}; provisional rows removed")
            return 1
        con.execute("UPDATE classifier_batches SET batch_id = ?, note = note || ?; " if False else
                    "UPDATE classifier_batches SET batch_id = ?, note = ? WHERE batch_id = ?",
                    (mb.id, f"{NOTE}; arm {arm}; codebook {cb}; max_run {mr:.0f}; processing_status {mb.processing_status}", provisional))
        con.execute("UPDATE classifier_calls SET batch_id = ? WHERE batch_id = ?", (mb.id, provisional))
        pilot["arms"][arm].setdefault("batches", []).append(mb.id)
        pilot["arms"][arm]["submitted_at"] = now
        save_pilot(pilot)
        log(f"{arm}: submitted {mb.id}: {len(reqs)} requests, prompt_version {pv}, status {mb.processing_status}")
    return 0


def collect(con):
    import anthropic
    client = anthropic.Anthropic()
    pilot = load_pilot()
    batches = [dict(r) for r in con.execute("SELECT * FROM classifier_batches WHERE status = ? ORDER BY created_at", (BATCH_STATUS,))]
    if not batches:
        log("no pilot batches pending")
        return 0
    pending = 0
    for b in batches:
        bid, model, pv = b["batch_id"], b["model"], b["prompt_version"]
        mb = client.messages.batches.retrieve(bid)
        if mb.processing_status != "ended":
            rcnt = mb.request_counts
            log(f"{bid}: {mb.processing_status} ({rcnt.processing} processing, {rcnt.succeeded} succeeded, {rcnt.errored} errored)")
            pending += 1
            continue
        ok = bad = flagged = 0
        cost_total = 0.0
        for result in client.messages.batches.results(bid):
            cid = result.custom_id                             # cbv2-<arm>-<vid>_c001
            arm = next(a for a in ARMS if cid.startswith(API_PREFIX + a + "-"))
            cid = CALL_PREFIX + cid[len(API_PREFIX + arm + "-"):]
            if result.result.type == "succeeded":
                msg = result.result.message
                res, usage = cl.parse_reply(msg), msg.usage
                cost = cl.usd(usage, model, batch=True) if usage else 0.0
                cost_total += cost
                if cl.apply_result(con, bid, model, pv, cid, res, usage, cost):
                    ok += 1
                    flagged += int(bool(res.get("contains_instructions")))
                else:
                    bad += 1
            else:
                bad += 1
                err = getattr(getattr(result.result, "error", None), "type", result.result.type)
                cl.apply_result(con, bid, model, pv, cid, None, None, 0.0, error=f"batch {result.result.type}: {err}")
        con.execute("UPDATE classifier_batches SET status = 'collected', n_succeeded = ?, n_errored = ?, cost_usd = ?, collected_at = ? "
                    "WHERE batch_id = ?", (ok, bad, round(cost_total, 4), state.now_iso(), bid))
        for arm, a in pilot["arms"].items():
            if bid in a.get("batches", []):
                a["actual_usd"] = round(a.get("actual_usd", 0.0) + cost_total, 4)
                a.update({"n_succeeded": ok, "n_errored": bad, "instruction_flags": flagged, "collected_at": state.now_iso()})
        log(f"{bid}: {ok} labels, {bad} errored/unparseable, ${cost_total:.4f}; {flagged} window(s) flagged instruction-like text")
    pilot["actual_total_usd"] = round(float(con.execute("SELECT COALESCE(SUM(cost_usd), 0) FROM classifier_calls WHERE video_id LIKE ?",
                                                        (CALL_PREFIX + "%",)).fetchone()[0]), 4)
    save_pilot(pilot)
    return 3 if pending else 0


# --- analyze ------------------------------------------------------------------------

def arm_episodes(cands, have):
    """Merged about_* spans (classify.merge_episodes, gap 90) and mention spans
    from the labelled candidate rows of one meeting."""
    rows = []
    for c in cands:
        r = have[c["id"]]
        if r is None:
            continue
        rows.append({"id": c["id"], "video_id": c["video_id"], "start_s": c["start_s"], "end_s": c["end_s"],
                     "label": r["label"], "rationale": r["rationale"], "family_rates": r["family_rates"]})
    located = cl.merge_episodes(rows, GAP, cl.ABOUT, "e")
    for e in located:
        e["tags"] = cl.TAGS.get(e["label"], "both")
    return rows, located


def fam_of_tags(tags):
    return {"density": tags in ("density", "both"), "transit": tags in ("transit", "both"), "any": True, "both_label": tags == "both"}


def analyze(con, arms):
    sample, usable, ref = load_ref()
    keywords = cl.keywords_for("en")
    pilot = load_pilot()
    model = model_id()
    vids = [m["video_id"] for m in usable]
    ref = [r for r in ref if r["video_id"] in vids]
    orig_missed = json.loads((ORIG / "results.json").read_text())["missed"]
    kw2_missed = json.loads((REF / "results.json").read_text())["missed"]
    res = {"at": state.now_iso(), "model": model, "keywords_version": keywords.version, "reference": str(REF),
           "meetings": vids, "n_ref_windows": len(ref), "ref_labels": dict(Counter(r["label"] for r in ref)),
           "codebook_versions": dict(cl.CODEBOOK_VERSIONS), "arms": {}}
    arm_rows = {}
    for arm in arms:
        cb, mr = ARMS[arm]
        per = arm_candidates(con, arm, usable, keywords)
        spans, cand_rows, unl = {}, {}, 0
        for vid, (cands, have) in per.items():
            unl += sum(1 for c in cands if have[c["id"]] is None)
            rows, located = arm_episodes(cands, have)
            spans[vid] = located
            cand_rows[vid] = rows
        all_rows = [r for v in cand_rows.values() for r in v]
        arm_rows[arm] = (cand_rows, spans)
        # per reference window: seconds inside the arm's located spans, tags
        for r in ref:
            vid = r["video_id"]
            eps = spans.get(vid, [])
            r[f"ep_{arm}"] = rc.covered([(e["start_s"], e["end_s"]) for e in eps], r["start_s"], r["end_s"])
            r[f"tags_{arm}"] = sorted({e["tags"] for e in eps if e["start_s"] < r["end_s"] and e["end_s"] > r["start_s"]})
            # the arm's own label on the candidate covering this minute (for the miss cause)
            lab = None
            for c in sorted(cand_rows.get(vid, []), key=lambda c: c["start_s"]):
                if c["start_s"] <= r["start_s"] < c["end_s"] or rc.covered([(c["start_s"], c["end_s"])], r["start_s"], r["end_s"]) >= HALF:
                    lab = c["label"]
                    break
            r[f"lab_{arm}"] = lab
        fams = {}
        for fam, labels in rc.FAMILIES.items():
            per_m = []
            for vid in vids:
                ab = [r for r in ref if r["video_id"] == vid and r["label"] in labels]
                cov = [r for r in ref if r["video_id"] == vid and r[f"ep_{arm}"] >= HALF
                       and any(t in rc.TAG_FAMILY[fam] for t in r[f"tags_{arm}"])]
                per_m.append({"video_id": vid, "about_min": len(ab) * W / 60, "located_min": sum(r[f"ep_{arm}"] for r in ab) / 60,
                              "found_w": sum(1 for r in ab if r[f"ep_{arm}"] >= HALF), "n_w": len(ab),
                              "cov_w": len(cov), "cov_about_w": sum(1 for r in cov if r["label"] in labels),
                              "located_total_min": sum(e["end_s"] - e["start_s"] for e in spans.get(vid, [])
                                                       if fam_of_tags(e["tags"])[fam]) / 60})
            tot = {k: sum(p[k] for p in per_m) for k in ("about_min", "located_min", "found_w", "n_w", "cov_w", "cov_about_w", "located_total_min")}
            fams[fam] = {"about_min": tot["about_min"], "located_min": round(tot["located_min"], 1),
                         "recall_min": round(tot["located_min"] / tot["about_min"], 3) if tot["about_min"] else None,
                         "recall_min_ci": rc.ratio_ci(per_m, "located_min", "about_min"),
                         "recall_window": round(tot["found_w"] / tot["n_w"], 3) if tot["n_w"] else None,
                         "recall_window_ci": rc.ratio_ci(per_m, "found_w", "n_w"),
                         "covered_w": tot["cov_w"], "covered_about_w": tot["cov_about_w"],
                         "precision_window": round(tot["cov_about_w"] / tot["cov_w"], 3) if tot["cov_w"] else None,
                         "precision_ci": rc.ratio_ci(per_m, "cov_about_w", "cov_w"),
                         "located_total_min": round(tot["located_total_min"], 1), "per_meeting": per_m}
        # misses: reference about-windows < 30 s inside a located span, with cause
        missed = []
        for r in ref:
            if r["label"] in cl.ABOUT and r[f"ep_{arm}"] < HALF:
                if r["screen_s"] < HALF:
                    why = "screen"
                elif r[f"lab_{arm}"] is None:
                    why = "unlabelled"
                elif r[f"lab_{arm}"] in cl.ABOUT:
                    why = "other"
                else:
                    why = "classifier"
                missed.append({"video_id": r["video_id"], "start": rc.hms(r["start_s"]), "start_s": r["start_s"], "label": r["label"],
                               "why": why, "arm_label": r[f"lab_{arm}"], "rationale": r.get("rationale"),
                               "text": " ".join((r["core_text"] or r["text"]).split()[:24])})
        # over-calls: candidate windows the arm labelled about_* whose reference minutes are all non-about
        over = []
        for vid, rows in cand_rows.items():
            for c in rows:
                if c["label"] not in cl.ABOUT:
                    continue
                refw = [r for r in ref if r["video_id"] == vid and r["start_s"] >= c["start_s"] and r["end_s"] <= c["end_s"]]
                if refw and not any(r["label"] in cl.ABOUT for r in refw):
                    over.append({"video_id": vid, "start": rc.hms(c["start_s"]), "end": rc.hms(c["end_s"]), "arm_label": c["label"],
                                 "rationale": c["rationale"], "ref_labels": [r["label"] for r in refw],
                                 "ref_rationales": [r.get("rationale") for r in refw],
                                 "kind": "possible over-call (ref mention_only)" if any(r["label"] == "mention_only" for r in refw) else "over-call (ref no)"})
        # the original 30 misses (pre-expansion) and the 27 post-expansion ones: recovered?
        def recovered(mlist):
            out = []
            for m in mlist:
                r = next((x for x in ref if x["video_id"] == m["video_id"] and x["start_s"] == m["start_s"]), None)
                out.append({"video_id": m["video_id"], "start": m["start"], "label": m["label"], "orig_why": m["why"],
                            "found": bool(r and r[f"ep_{arm}"] >= HALF), "arm_label": r[f"lab_{arm}"] if r else None})
            return out
        rec_orig = recovered(orig_missed)
        rec_kw2 = recovered(kw2_missed)
        lab_counts = dict(Counter(r["label"] for r in all_rows))
        a = pilot["arms"].get(arm, {})
        res["arms"][arm] = {"codebook": cb, "codebook_version": cl.CODEBOOK_VERSIONS[cb], "max_run_s": mr, "prompt_version": pv_for(arm, model),
                            "n_candidates": len(all_rows) + unl, "n_unlabelled": unl, "labels": lab_counts,
                            "n_located_episodes": sum(len(v) for v in spans.values()),
                            "located_total_min": round(sum(e["end_s"] - e["start_s"] for v in spans.values() for e in v) / 60, 1),
                            "est_usd": a.get("est_usd"), "actual_usd": a.get("actual_usd"), "batches": a.get("batches"),
                            "instruction_flags": a.get("instruction_flags"),
                            "families": fams, "missed": missed, "miss_share": dict(Counter(m["why"] for m in missed)),
                            "overcalls": over, "recovered_of_30_original": sum(1 for x in rec_orig if x["found"]),
                            "recovered_of_27_post_kw2": sum(1 for x in rec_kw2 if x["found"]),
                            "recovered_original_detail": rec_orig, "recovered_post_kw2_detail": rec_kw2}
        (OUT / f"episodes_{arm}.json").write_text(json.dumps(
            {vid: [{k: v for k, v in e.items() if k != "windows"} | {"windows": e["windows"]} for e in eps] for vid, eps in spans.items()},
            indent=1, ensure_ascii=False))
        (OUT / f"windows_{arm}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in all_rows))
        log(f"{arm}: {len(all_rows)} labelled candidates ({unl} unlabelled) {lab_counts}; {res['arms'][arm]['n_located_episodes']} located spans = "
            f"{res['arms'][arm]['located_total_min']} min; recall any {fams['any']['recall_min']} transit {fams['transit']['recall_min']} "
            f"density {fams['density']['recall_min']}; precision any {fams['any']['precision_window']}; misses {res['arms'][arm]['miss_share']}; "
            f"over-calls {len(over)}; recovered {res['arms'][arm]['recovered_of_30_original']}/30 original, "
            f"{res['arms'][arm]['recovered_of_27_post_kw2']}/27 post-kw2")
    (OUT / "results.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    (OUT / "windows_reference_with_arms.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in ref))
    write_markdown(res, arms)
    return 0


def write_markdown(res, arms):
    L = ["| family | arm | about-min | located-min | recall (min) | 95 % CI | window recall | covered-w | precision | 95 % CI | located total (min) |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for fam in ("density", "transit", "any", "both_label"):
        for arm in arms:
            f = res["arms"][arm]["families"][fam]
            ci, pci = f["recall_min_ci"], f["precision_ci"]
            fmt = lambda x: "-" if x is None else f"{x:.3f}"
            L.append(f"| {fam} | {arm} | {f['about_min']:.0f} | {f['located_min']:.1f} | {fmt(f['recall_min'])} | "
                     f"{fmt(ci[0])}-{fmt(ci[1])} | {fmt(f['recall_window'])} | {f['covered_w']} | {fmt(f['precision_window'])} | "
                     f"{fmt(pci[0])}-{fmt(pci[1])} | {f['located_total_min']:.1f} |")
    L += ["", "| arm | codebook | max-run | prompt_version | candidates | labels | located spans (min) | est $ | actual $ | misses (cause) | over-calls | recovered of 30 / 27 |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for arm in arms:
        a = res["arms"][arm]
        L.append(f"| {arm} | {a['codebook_version']} | {a['max_run_s']:.0f} s | `{a['prompt_version']}` | {a['n_candidates']} | {a['labels']} | "
                 f"{a['n_located_episodes']} ({a['located_total_min']}) | {a['est_usd'] if a['est_usd'] is not None else '-'} | "
                 f"{a['actual_usd'] if a['actual_usd'] is not None else '0 (pipeline rows)'} | {a['miss_share']} | {len(a['overcalls'])} | "
                 f"{a['recovered_of_30_original']} / {a['recovered_of_27_post_kw2']} |")
    (OUT / "table.md").write_text("\n".join(L) + "\n")
    # over-calls, per arm
    M = ["# Over-calls: candidate windows labelled about_* whose reference 60-s windows are all mention_only / no", "",
         "The reference is the pinned codebook on 60-s windows read without keyword context; 'possible over-call' means the reference "
         "called at least one of the minutes mention_only (a judgment call for Puck), 'over-call (ref no)' means the reference saw no "
         "topic at all.", ""]
    for arm in arms:
        a = res["arms"][arm]
        M += [f"## {arm} ({a['codebook_version']}, {a['max_run_s']:.0f} s) -- {len(a['overcalls'])}", ""]
        for o in a["overcalls"]:
            M.append(f"- **{o['video_id']} {o['start']}-{o['end']}** {o['arm_label']} [{o['kind']}] {o['rationale']} -- reference: "
                     + "; ".join(f"{l}: {r}" for l, r in zip(o["ref_labels"], o["ref_rationales"])))
        M.append("")
    (OUT / "overcalls.md").write_text("\n".join(M))
    for arm in arms:
        a = res["arms"][arm]
        X = [f"# {arm}: remaining missed about-windows ({len(a['missed'])} of {sum(1 for _ in a['missed']) and res['ref_labels'].get('about_density', 0) + res['ref_labels'].get('about_transit', 0) + res['ref_labels'].get('about_both', 0)} about-minutes)", "",
             "why = screen: never a candidate on the Mac-ASR locate track (same in every arm); classifier: a candidate this arm labelled "
             "mention_only / no; other: labelled about but outside the merged span.", ""]
        for m in a["missed"]:
            X.append(f"- **{m['video_id']} {m['start']}** {m['label']} [{m['why']}; arm {m['arm_label'] or '-'}] {m['rationale'] or ''} -- \"{m['text']}\"")
        X += ["", "## The 30 original misses (pre-expansion run)", ""]
        for x in a["recovered_original_detail"]:
            X.append(f"- {x['video_id']} {x['start']} {x['label']} (was: {x['orig_why']}) -> {'FOUND' if x['found'] else 'missed'}"
                     + (f" (arm label {x['arm_label']})" if x["arm_label"] else ""))
        (OUT / f"missed_{arm}.md").write_text("\n".join(X) + "\n")
    # codebook diff
    d = difflib.unified_diff(cl.INSTRUCTIONS.splitlines(), cl.INSTRUCTIONS_V2.splitlines(), "INSTRUCTIONS (v1, t0-a4a8e55760)",
                             f"INSTRUCTIONS_V2 ({cl.CODEBOOK_VERSIONS['v2']})", lineterm="", n=1)
    (OUT / "codebook_diff.md").write_text("```diff\n" + "\n".join(d) + "\n```\n")
    print((OUT / "table.md").read_text())


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["estimate", "submit", "collect", "analyze"])
    ap.add_argument("--arms", default="v1_240,v1_120,v2_120")
    ap.add_argument("--cap-usd", type=float, default=6.0)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--state", default=None)
    a = ap.parse_args(argv)
    arms = [x.strip() for x in a.arms.split(",") if x.strip()]
    for x in arms:
        if x not in ARMS:
            ap.error(f"unknown arm {x}; choose from {', '.join(ARMS)}")
    con = state.connect(a.state)
    OUT.mkdir(parents=True, exist_ok=True)
    if a.cmd == "estimate":
        estimate(con, arms)
        return 0
    if a.cmd == "submit":
        return submit(con, arms, a.cap_usd, a.dry_run)
    if a.cmd == "collect":
        return collect(con)
    return analyze(con, arms)


if __name__ == "__main__":
    sys.exit(main())
