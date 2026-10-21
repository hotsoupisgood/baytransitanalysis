#!/usr/bin/env python3
"""
scripts/codebook_v21_check.py -- the $0.25 check of codebook v2.1 on the pilot's
111 windows (plan/CODEBOOK-V2-PILOT.md 9 caveat -> 10; 2026-10-18).

v2.1 = v2 + Puck's one-sentence transit rule (classify.INSTRUCTIONS_V21,
CODEBOOK_VERSIONS['v2.1']). It is read on EXACTLY the 111 v2@120 pilot windows
(output/validation/codebook_v2/windows_v2_120.jsonl: ids and bounds; the text is
rebuilt from the same Mac-ASR locate track with the same +-30 s pad), not on a
re-screen -- the keyword list changed after the pilot (kw-24e8a4acd1 -> the
current list gives 126 windows), and the question is whether the SENTENCE moves
labels, so the window set is held fixed.

Then: (1) label by label, v2 -> v2.1, each change judged against the adjudicated
reference (adjudicated/reference_adjudicated.jsonl: the 60-s minutes the window
covers) as toward / away / neutral; (2) recall and precision per family for
v2.1 on the adjudicated reference with the same minute-coverage rule and
4-cluster bootstrap as codebook_v2_adjudicate.py, beside v2@120's numbers.

Rows: classifier_calls under video_id 'cbv21:<meeting>', prompt_version
'v2.1-<hash>/claude-sonnet-4-6/mr120'; batch row status 'codebook_v21_check',
note 'pilot-v2.1'. No episodes row, no v1 / v2 row touched.

    submit [--cap-usd 1] [--dry-run]
    collect
    analyze
"""
import argparse, json, sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))
sys.path.insert(0, str(ROOT / "scripts"))
import state                # noqa: E402
import segment as sg        # noqa: E402
import classify as cl       # noqa: E402
import recall_check as rc   # noqa: E402

PILOT = ROOT / "output" / "validation" / "codebook_v2"
ADJ = PILOT / "adjudicated"
OUT = PILOT / "v21_check"
REF = ROOT / "output" / "validation" / "recall" / "post_kw2"
CALL_PREFIX, API_PREFIX = "cbv21:", "cbv21-"
BATCH_STATUS, NOTE = "codebook_v21_check", "pilot-v2.1: 111-window check of codebook v2.1 (plan/CODEBOOK-V2-PILOT.md 10)"
CODEBOOK, MAX_RUN, PAD, GAP, HALF, W = "v2.1", 120.0, 30.0, 90.0, 30.0, 60.0
MODEL = cl.LEGACY_MODEL
ABOUT = cl.ABOUT
FAM_OF = {"about_density": "density", "about_transit": "transit", "about_both": "both"}


def log(msg):
    print(f"{state.now_iso()} {msg}", flush=True)


def pv():
    return cl.prompt_version_for(MODEL, cl.DEFAULT_LANG, "asr_local", CODEBOOK, MAX_RUN)


def v2_windows():
    return [json.loads(l) for l in (PILOT / "windows_v2_120.jsonl").read_text().splitlines() if l.strip()]


def tracks():
    s = json.loads((REF / "sample.json").read_text())
    return {m["video_id"]: m["vtt"] for m in s["meetings"] if m.get("ingest_status") == "done"}


def with_text(rows):
    segs = {vid: sg.read_segments(p) for vid, p in tracks().items()}
    for r in rows:
        ss = segs[r["video_id"]]
        end = max((x["end"] for x in ss), default=0)
        r["pad_start_s"], r["pad_end_s"] = max(0.0, r["start_s"] - PAD), min(end, r["end_s"] + PAD)
        r["text"] = cl.window_text(ss, r["pad_start_s"], r["pad_end_s"])
        r["words"] = len(r["text"].split())
        r["id21"] = CALL_PREFIX + r["id"][len("cbv2:"):]
    return rows


def submit(con, cap_usd, dry_run=False):
    import spend                                                   # noqa: E402
    rows = with_text(v2_windows())
    p = pv()
    have = {r["id"] for r in con.execute("SELECT id FROM classifier_calls WHERE video_id LIKE ? AND prompt_version = ? AND label IS NOT NULL", (CALL_PREFIX + "%", p))}
    pend = {r["id"] for r in con.execute("SELECT id FROM classifier_calls WHERE video_id LIKE ? AND prompt_version = ? AND label IS NULL AND batch_id LIKE 'msgbatch%'", (CALL_PREFIX + "%", p))}
    send = [r for r in rows if r["id21"] not in have and r["id21"] not in pend]
    tokens = sum(cl.caption_tokens_for(r["words"]) for r in send)
    est = cl.estimate_usd(tokens, len(send), MODEL, batch=True, codebook=CODEBOOK)
    worst = sum(cl.worst_case_usd(r["text"], MODEL, batch=True, codebook=CODEBOOK) for r in send)
    log(f"{len(rows)} pilot windows, {len(send)} to send ({len(have)} labelled, {len(pend)} pending); codebook {cl.codebook_version_for(CODEBOOK)}, "
        f"prompt_version {p}; estimate ${est:.3f}, worst ${worst:.3f}, cap ${cap_usd}")
    if est > cap_usd:
        log("REFUSED: over the ceiling"); return 2
    why = spend.claude_refusal(con, worst)
    if why:
        log(f"REFUSED: {why}"); return 2
    if not send:
        return 0
    if dry_run:
        log("dry run"); return 0
    import anthropic
    client = anthropic.Anthropic()
    now = state.now_iso()
    provisional = f"pending-cbv21-{now}"
    con.execute("INSERT INTO classifier_batches (batch_id, created_at, status, model, prompt_version, n_requests, est_usd, note) VALUES (?,?,?,?,?,?,?,?)",
                (provisional, now, BATCH_STATUS, MODEL, p, len(send), round(est, 4), NOTE))
    reqs = []
    for r in send:
        c = {"id": r["id21"], "start_s": r["start_s"], "end_s": r["end_s"], "pad_start_s": r["pad_start_s"], "pad_end_s": r["pad_end_s"],
             "words": r["words"], "hits_per_100w": None, "family_rates": r.get("family_rates") or {}, "top_keywords": [], "keywords_version": "kw-24e8a4acd1"}
        if isinstance(c["family_rates"], str):
            c["family_rates"] = json.loads(c["family_rates"] or "{}")
        rec = cl.call_record(c, CALL_PREFIX + r["video_id"], MODEL, p, batch_id=provisional)
        cl.insert_call(con, cl.call_row(rec), replace=True)
        reqs.append({"custom_id": API_PREFIX + r["id21"][len(CALL_PREFIX):], "params": cl.request_params(MODEL, r["text"], cl.DEFAULT_LANG, codebook=CODEBOOK)})
    try:
        mb = client.messages.batches.create(requests=reqs)
    except Exception as e:                                         # noqa: BLE001
        con.execute("DELETE FROM classifier_calls WHERE batch_id = ? AND label IS NULL", (provisional,))
        con.execute("DELETE FROM classifier_batches WHERE batch_id = ?", (provisional,))
        log(f"batch create failed: {type(e).__name__}: {str(e)[:200]}"); return 1
    con.execute("UPDATE classifier_batches SET batch_id = ?, note = ? WHERE batch_id = ?", (mb.id, f"{NOTE}; processing_status {mb.processing_status}", provisional))
    con.execute("UPDATE classifier_calls SET batch_id = ? WHERE batch_id = ?", (mb.id, provisional))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "batch.json").write_text(json.dumps({"batch_id": mb.id, "submitted_at": now, "n": len(reqs), "codebook_version": cl.codebook_version_for(CODEBOOK),
                                                "prompt_version": p, "est_usd": round(est, 4), "worst_usd": round(worst, 4)}, indent=1))
    log(f"submitted {mb.id}: {len(reqs)} requests")
    return 0


def collect(con):
    import anthropic
    client = anthropic.Anthropic()
    for b in [dict(r) for r in con.execute("SELECT * FROM classifier_batches WHERE status = ?", (BATCH_STATUS,))]:
        bid = b["batch_id"]
        mb = client.messages.batches.retrieve(bid)
        if mb.processing_status != "ended":
            rcnt = mb.request_counts
            log(f"{bid}: {mb.processing_status} ({rcnt.processing} processing, {rcnt.succeeded} succeeded)"); continue
        ok = bad = 0
        cost = 0.0
        for result in client.messages.batches.results(bid):
            cid = CALL_PREFIX + result.custom_id[len(API_PREFIX):]
            if result.result.type == "succeeded":
                msg = result.result.message
                res, usage = cl.parse_reply(msg), msg.usage
                c = cl.usd(usage, b["model"], batch=True) if usage else 0.0
                cost += c
                ok += int(bool(cl.apply_result(con, bid, b["model"], b["prompt_version"], cid, res, usage, c)))
            else:
                bad += 1
                cl.apply_result(con, bid, b["model"], b["prompt_version"], cid, None, None, 0.0, error=f"batch {result.result.type}")
        con.execute("UPDATE classifier_batches SET status = 'collected', n_succeeded = ?, n_errored = ?, cost_usd = ?, collected_at = ? WHERE batch_id = ?",
                    (ok, bad, round(cost, 4), state.now_iso(), bid))
        meta = json.loads((OUT / "batch.json").read_text()) if (OUT / "batch.json").exists() else {}
        meta.update({"collected_at": state.now_iso(), "n_succeeded": ok, "n_errored": bad, "actual_usd": round(cost, 4)})
        (OUT / "batch.json").write_text(json.dumps(meta, indent=1))
        log(f"{bid}: {ok} labels, {bad} errored, ${cost:.4f}")
    return 0


def spans_of(rows, label_key):
    """Merged about_* spans per meeting (gap 90 s, core bounds), as the pipeline merges."""
    xs = [dict(r, label=r[label_key], rationale="", family_rates={}) for r in rows if r.get(label_key)]
    eps = cl.merge_episodes(xs, GAP, ABOUT, "x")
    by = {}
    for e in eps:
        by.setdefault(e["video_id"], []).append((e["start_s"], e["end_s"], cl.TAGS[e["label"]]))
    return by


def score(ref, spans, vids):
    fams = {}
    for fam, labels in rc.FAMILIES.items():
        per = []
        for vid in vids:
            rows = [r for r in ref if r["video_id"] == vid]
            sp = spans.get(vid, [])
            cov = lambda r: rc.covered([(a, b) for a, b, _ in sp], r["start_s"], r["end_s"])   # noqa: E731
            tags = lambda r: {t for a, b, t in sp if a < r["end_s"] and b > r["start_s"]}      # noqa: E731
            ab = [r for r in rows if r["label"] in labels]
            inside = [r for r in rows if cov(r) >= HALF and any(t in rc.TAG_FAMILY[fam] for t in tags(r))]
            per.append({"video_id": vid, "about_min": len(ab) * W / 60, "located_min": sum(cov(r) for r in ab) / 60,
                        "cov_w": len(inside), "cov_about_w": sum(1 for r in inside if r["label"] in labels)})
        tot = {k: sum(p[k] for p in per) for k in ("about_min", "located_min", "cov_w", "cov_about_w")}
        fams[fam] = {"about_min": tot["about_min"], "located_min": round(tot["located_min"], 1),
                     "recall": round(tot["located_min"] / tot["about_min"], 3) if tot["about_min"] else None,
                     "recall_ci": rc.ratio_ci(per, "located_min", "about_min"),
                     "precision": round(tot["cov_about_w"] / tot["cov_w"], 3) if tot["cov_w"] else None,
                     "precision_ci": rc.ratio_ci(per, "cov_about_w", "cov_w"), "covered_w": tot["cov_w"]}
    return fams


def analyze(con):
    OUT.mkdir(parents=True, exist_ok=True)
    rows = v2_windows()
    p = pv()
    lab21 = {r["id"]: dict(r) for r in con.execute("SELECT id, label, rationale, cost_usd FROM classifier_calls WHERE video_id LIKE ? AND prompt_version = ? AND label IS NOT NULL",
                                                    (CALL_PREFIX + "%", p))}
    for r in rows:
        x = lab21.get(CALL_PREFIX + r["id"][len("cbv2:"):])
        r["label_v2"], r["label_v21"] = r["label"], (x or {}).get("label")
        r["rationale_v21"] = (x or {}).get("rationale")
    unl = [r for r in rows if not r["label_v21"]]
    if unl:
        log(f"WARNING: {len(unl)} windows unlabelled under v2.1")
    ref = [json.loads(l) for l in (ADJ / "reference_adjudicated.jsonl").read_text().splitlines()[1:] if l.strip()]
    vids = sorted({r["video_id"] for r in ref})
    # 1. per-window changes judged against the adjudicated minutes the window covers
    changes = []
    for r in rows:
        if not r["label_v21"] or r["label_v21"] == r["label_v2"]:
            continue
        mins = [m for m in ref if m["video_id"] == r["video_id"] and m["start_s"] >= r["start_s"] - 1 and m["end_s"] <= r["end_s"] + 1]
        ref_labels = [m["label"] for m in mins]
        ref_about = any(l in ABOUT for l in ref_labels)
        ref_fams = {FAM_OF[l] for l in ref_labels if l in ABOUT}
        def agree(lab):
            if lab in ABOUT:
                return 2 if (ref_about and (FAM_OF[lab] in ref_fams or FAM_OF[lab] == "both" or "both" in ref_fams)) else (1 if ref_about else 0)
            return 2 if not ref_about else 0
        a2, a21 = agree(r["label_v2"]), agree(r["label_v21"])
        verdict = "toward" if a21 > a2 else ("away" if a21 < a2 else "neutral")
        changes.append({"id": r["id"], "video_id": r["video_id"], "start": rc.hms(r["start_s"]), "v2": r["label_v2"], "v21": r["label_v21"],
                        "reference": ref_labels, "verdict": verdict, "v21_rationale": r["rationale_v21"]})
    # 2. recall / precision on the adjudicated reference, v2 vs v2.1, same windows
    s2 = score(ref, spans_of(rows, "label_v2"), vids)
    s21 = score(ref, spans_of(rows, "label_v21"), vids)
    cost = float(con.execute("SELECT COALESCE(SUM(cost_usd), 0) FROM classifier_calls WHERE video_id LIKE ? AND prompt_version = ?", (CALL_PREFIX + "%", p)).fetchone()[0])
    res = {"at": state.now_iso(), "codebook_version": cl.codebook_version_for(CODEBOOK), "prompt_version": p, "n_windows": len(rows), "n_unlabelled": len(unl),
           "labels_v2": dict(Counter(r["label_v2"] for r in rows)), "labels_v21": dict(Counter(r["label_v21"] for r in rows)),
           "n_changed": len(changes), "verdicts": dict(Counter(c["verdict"] for c in changes)), "changes": changes,
           "transitions": dict(Counter(f"{c['v2']}->{c['v21']}" for c in changes)),
           "v2_120": s2, "v21_120": s21, "cost_usd": round(cost, 4)}
    (OUT / "results.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    f3 = lambda x: "—" if x is None else f"{x:.3f}"     # noqa: E731
    L = [f"codebook v2.1 `{res['codebook_version']}` on the 111 v2@120 pilot windows; ${cost:.4f}; labels v2 {res['labels_v2']} -> v2.1 {res['labels_v21']}",
         f"changed windows: {len(changes)} ({res['verdicts']}); transitions {res['transitions']}", "",
         "| family | arm | about-min | recall | 95 % CI | precision | 95 % CI |", "|---|---|---|---|---|---|---|"]
    for fam in ("density", "transit", "any", "both_label"):
        for arm, s in (("v2_120", s2), ("v21_120", s21)):
            f = s[fam]
            L.append(f"| {fam} | {arm} | {f['about_min']:.0f} | {f3(f['recall'])} | {f3(f['recall_ci'][0])}–{f3(f['recall_ci'][1])} | "
                     f"{f3(f['precision'])} | {f3(f['precision_ci'][0])}–{f3(f['precision_ci'][1])} |")
    L += ["", "| window | v2 | v2.1 | adjudicated reference (2 min) | verdict | v2.1 rationale |", "|---|---|---|---|---|---|"]
    for c in changes:
        L.append(f"| {c['video_id']} {c['start']} | {c['v2']} | {c['v21']} | {'/'.join(c['reference'])} | **{c['verdict']}** | {c['v21_rationale']} |")
    (OUT / "table.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["submit", "collect", "analyze"])
    ap.add_argument("--cap-usd", type=float, default=1.0)
    ap.add_argument("--codebook", default=None, help="replicate arm: re-read the same 111 windows under another codebook (e.g. v2, to measure run-to-run noise)")
    ap.add_argument("--prefix", default=None, help="call-id / video_id prefix for a replicate arm (e.g. cbv2rep:)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--state", default=None)
    a = ap.parse_args(argv)
    global CODEBOOK, CALL_PREFIX, API_PREFIX, BATCH_STATUS, OUT
    if a.codebook:
        CODEBOOK = a.codebook
        CALL_PREFIX = a.prefix or f"cb{a.codebook.replace('.', '')}rep:"
        API_PREFIX = CALL_PREFIX[:-1] + "-"
        BATCH_STATUS = f"codebook_{a.codebook.replace('.', '')}_replicate"
        OUT = PILOT / f"{a.codebook.replace('.', '')}_replicate"
    con = state.connect(a.state)
    if a.cmd == "submit":
        return submit(con, a.cap_usd, a.dry_run)
    if a.cmd == "collect":
        return collect(con)
    return analyze(con)


if __name__ == "__main__":
    sys.exit(main())
