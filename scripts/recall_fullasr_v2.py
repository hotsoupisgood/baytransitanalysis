#!/usr/bin/env python3
"""
scripts/recall_fullasr_v2.py -- the "both of the above" pilot arm
(plan/CODEBOOK-V2-PILOT-FULLASR.md, 2026-10-18): codebook v2 with NO keyword
screen over EVERY 120-s window of the whole-meeting WhisperX transcripts of the
locate-recall sample (output/validation/recall/whisper/), scored for recall and
precision against the exhaustive 60-s reference labels (codebook v1, same
transcripts, output/validation/recall/windows.jsonl).

Nothing here touches the corpus, the pinned codebook, or any existing
classifier_calls / episodes row: calls are written under video_id
'recallv2:<meeting>' with v2's prompt_version, the batch row is status
'pilot_fullasr' (so recall_check.collect ignores it) and note 'pilot-fullasr'.
Episodes are merged in memory only (classify.merge_episodes, gap 90 s), never
written to `episodes`.

    windows                   120-s windows (+/- 30 s pad) of every transcript, reference labels attached, estimate
    submit [--cap-usd 4]      one Message Batch, codebook v2, prompt_version v2
    collect                   write labels back (DB + windows.jsonl)
    analyze                   recall / precision per family, 6-cluster bootstrap, over-calls, results.json
"""
import argparse, json, sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))
sys.path.insert(0, str(ROOT / "scripts"))
import state                 # noqa: E402
import segment as sg         # noqa: E402
import classify as cl        # noqa: E402
import recall_check as rc    # noqa: E402

REF = ROOT / "output" / "validation" / "recall"           # reference: whole-meeting transcripts + v1 60-s labels
OUT = ROOT / "output" / "validation" / "codebook_v2_fullasr"
WINDOW, PAD, GAP = 120.0, cl.DEFAULTS["pad"], cl.DEFAULTS["gap"]
MIN_WORDS = 10
CALL_PREFIX = "recallv2:"
API_ID_PREFIX = "recallv2-"
BATCH_STATUS = "pilot_fullasr"
NOTE = "pilot-fullasr"
MODEL = cl.LEGACY_MODEL
FAMILIES = rc.FAMILIES
TAG_FAMILY = rc.TAG_FAMILY
log = rc.log


# --- codebook v2 (owned by the sibling pilot; read only) ---------------------------------

CODEBOOK = "v2"


def codebook_v2():
    """(text, version) of codebook v2 exactly as pipeline/classify.py exposes
    it (CODEBOOKS / CODEBOOK_VERSIONS, `--codebook v2`)."""
    if CODEBOOK not in getattr(cl, "CODEBOOKS", {}):
        sys.exit("codebook v2 not found in pipeline/classify.py (CODEBOOKS['v2'])")
    return cl.codebook_text(CODEBOOK), cl.codebook_version_for(CODEBOOK)


def prompt_version_v2(ver=None):
    """v2's instrument id for this arm: the pipeline's own naming, with the
    120-s granularity named ('v2-<hash>/claude-sonnet-4-6/mr120')."""
    return cl.prompt_version_for(MODEL, cl.DEFAULT_LANG, None, codebook=CODEBOOK, max_run=WINDOW)


def request_params_v2(text, book=None):
    """The pipeline's request for a v2 window, byte-for-byte."""
    return cl.request_params(MODEL, text, cl.DEFAULT_LANG, codebook=CODEBOOK)


# --- windows -------------------------------------------------------------------------

def windows_path():
    return OUT / "windows.jsonl"


def read_windows():
    return [json.loads(l) for l in windows_path().read_text().splitlines() if l.strip()]


def write_windows(rows):
    windows_path().write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


def reference():
    """Reference 60-s windows with their v1 labels, by meeting."""
    by = defaultdict(list)
    for r in (json.loads(l) for l in (REF / "windows.jsonl").read_text().splitlines() if l.strip()):
        by[r["video_id"]].append(r)
    return by


def usable(m):
    """A meeting whose whole-meeting WhisperX transcript passed QA. gr-richmond-6033
    and -6038 failed (coverage 5 %, decoder loops; RECALL-CHECK.md) and are excluded
    from this arm: nothing to label, nothing to score."""
    return bool(m.get("transcript")) and m.get("ingest_status") == "done"


def build_windows():
    s = json.loads((REF / "sample.json").read_text())
    ref = reference()
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for m in s["meetings"]:
        if not usable(m):
            log(f"  {m['video_id']}: EXCLUDED ({m.get('ingest_status')}: {m.get('ingest_note') or 'no transcript'})")
            continue
        vid = m["video_id"]
        doc = json.loads(Path(m["transcript"]).read_text())
        segs = [{"start": float(x["start"]), "end": float(x["end"]), "text": x["text"]} for x in doc["segments"]]
        end_of_talk = max((x["end"] for x in segs), default=0)
        refs = {r["i"]: r for r in ref.get(vid, [])}
        n = int(end_of_talk // WINDOW) + 1
        for i in range(n):
            a, b = i * WINDOW, (i + 1) * WINDOW
            pa, pb = max(0.0, a - PAD), min(end_of_talk, b + PAD)
            core = cl.window_text(segs, a, b)
            text = cl.window_text(segs, pa, pb)
            # the two reference minutes this window covers
            rl = [refs[j]["label"] for j in (2 * i, 2 * i + 1) if j in refs]
            rows.append({"id": f"{CALL_PREFIX}{vid}_w{i:04d}", "video_id": vid, "body": m["body"], "i": i, "start_s": a, "end_s": b,
                         "pad_start_s": pa, "pad_end_s": pb, "words": len(text.split()), "core_words": len(core.split()),
                         "ref_labels": rl, "text": text, "core_text": core, "label": None})
        log(f"  {vid}: {n} windows of {WINDOW:.0f} s, {sum(1 for r in rows if r['video_id'] == vid and r['words'] >= MIN_WORDS)} with >= {MIN_WORDS} padded words")
    write_windows(rows)
    send = [r for r in rows if r["words"] >= MIN_WORDS]
    est, worst, book_tok = estimate(send)
    log(f"{len(rows)} windows ({len(send)} to send, {sum(r['words'] for r in send):,} padded words, "
        f"~{sum(cl.caption_tokens_for(r['words']) for r in send):,} caption tokens); codebook ~{book_tok:.0f} tokens; "
        f"estimate ${est:.2f} (worst case ${worst:.2f}) at the batch rate -> {windows_path()}")
    (OUT / "estimate.json").write_text(json.dumps({"at": state.now_iso(), "n_windows": len(rows), "n_to_send": len(send),
                                                   "padded_words": sum(r["words"] for r in send), "codebook_tokens": round(book_tok),
                                                   "est_usd": round(est, 3), "worst_usd": round(worst, 3), "model": MODEL}, indent=1))


def estimate(send):
    """classify.estimate_usd / worst_case_usd with the codebook token count
    scaled to v2's length when v2 is importable (else v1 length + 15 %)."""
    book_tok = cl.codebook_tokens(CODEBOOK)
    tokens = sum(cl.caption_tokens_for(r["words"]) for r in send)
    est = cl.estimate_usd(tokens, len(send), MODEL, batch=True, codebook=CODEBOOK)
    worst = sum(cl.worst_case_usd(r["text"], MODEL, batch=True, codebook=CODEBOOK) for r in send)
    return est, worst, book_tok


# --- submit / collect ------------------------------------------------------------------

def submit(con, cap_usd, dry_run=False):
    import spend                                               # noqa: E402
    book, ver = codebook_v2()
    pv = prompt_version_v2(ver)
    rows = read_windows()
    done = {r["id"] for r in con.execute("SELECT id FROM classifier_calls WHERE video_id LIKE ? AND label IS NOT NULL", (CALL_PREFIX + "%",))}
    pending = {r["id"] for r in con.execute("SELECT id FROM classifier_calls WHERE video_id LIKE ? AND label IS NULL AND batch_id LIKE 'msgbatch%'",
                                            (CALL_PREFIX + "%",))}
    send = [r for r in rows if r["words"] >= MIN_WORDS and r["id"] not in done and r["id"] not in pending]
    est, worst, book_tok = estimate(send)
    log(f"submit: {len(send)} windows ({len(done)} labelled, {len(pending)} pending), model {MODEL}, codebook {ver} "
        f"(~{book_tok:.0f} tokens), prompt_version {pv}, estimate ${est:.2f}, worst ${worst:.2f}, cap ${cap_usd}")
    if est > cap_usd:
        log(f"REFUSED: estimate ${est:.2f} exceeds the ${cap_usd} ceiling for this arm")
        return 2
    why = spend.claude_refusal(con, worst)
    if why:
        log(f"REFUSED: {why}")
        return 2
    if not send:
        log("nothing to submit")
        return 0
    if dry_run:
        log("dry run: nothing submitted")
        return 0
    import anthropic
    client = anthropic.Anthropic()
    now = state.now_iso()
    provisional = f"pending-recallv2-{now}"
    con.execute("INSERT INTO classifier_batches (batch_id, created_at, status, model, prompt_version, n_requests, est_usd, note) "
                "VALUES (?,?,?,?,?,?,?,?)", (provisional, now, BATCH_STATUS, MODEL, pv, len(send), round(est, 4), NOTE))
    reqs = []
    for r in send:
        c = {"id": r["id"], "start_s": r["start_s"], "end_s": r["end_s"], "pad_start_s": r["pad_start_s"], "pad_end_s": r["pad_end_s"],
             "words": r["words"], "hits_per_100w": None, "family_rates": {}, "top_keywords": []}
        rec = cl.call_record(c, CALL_PREFIX + r["video_id"], MODEL, pv, batch_id=provisional)
        cl.insert_call(con, cl.call_row(rec), replace=True)
        reqs.append({"custom_id": API_ID_PREFIX + r["id"][len(CALL_PREFIX):], "params": request_params_v2(r["text"], book)})
    try:
        mb = client.messages.batches.create(requests=reqs)
    except Exception as e:                                     # noqa: BLE001 -- nothing billed; undo
        con.execute("DELETE FROM classifier_calls WHERE batch_id = ? AND label IS NULL", (provisional,))
        con.execute("DELETE FROM classifier_batches WHERE batch_id = ?", (provisional,))
        log(f"batch create failed: {type(e).__name__}: {str(e)[:200]}; provisional rows removed")
        return 1
    con.execute("UPDATE classifier_batches SET batch_id = ?, note = ? WHERE batch_id = ?",
                (mb.id, f"{NOTE}; codebook {ver}; processing_status {mb.processing_status}", provisional))
    con.execute("UPDATE classifier_calls SET batch_id = ? WHERE batch_id = ?", (mb.id, provisional))
    meta = {"batch_id": mb.id, "submitted_at": now, "n_requests": len(reqs), "codebook_version": ver, "prompt_version": pv,
            "model": MODEL, "est_usd": round(est, 4), "worst_usd": round(worst, 4), "codebook_chars": len(book)}
    (OUT / "batch.json").write_text(json.dumps(meta, indent=1))
    log(f"submitted {mb.id}: {len(reqs)} requests, status {mb.processing_status}")
    return 0


def collect(con):
    import anthropic
    client = anthropic.Anthropic()
    batches = [dict(r) for r in con.execute("SELECT * FROM classifier_batches WHERE status = ? ORDER BY created_at", (BATCH_STATUS,))]
    if not batches:
        log("no pilot_fullasr batches pending")
        return 0
    rows = {r["id"]: r for r in read_windows()}
    for b in batches:
        bid, model, pv = b["batch_id"], b["model"], b["prompt_version"]
        mb = client.messages.batches.retrieve(bid)
        if mb.processing_status != "ended":
            rcnt = mb.request_counts
            log(f"{bid}: {mb.processing_status} ({rcnt.processing} processing, {rcnt.succeeded} succeeded, {rcnt.errored} errored)")
            continue
        ok = bad = flagged = 0
        cost_total = 0.0
        tok = Counter()
        for result in client.messages.batches.results(bid):
            cid = result.custom_id
            if cid.startswith(API_ID_PREFIX):
                cid = CALL_PREFIX + cid[len(API_ID_PREFIX):]
            if result.result.type == "succeeded":
                msg = result.result.message
                res, usage = cl.parse_reply(msg), msg.usage
                cost = cl.usd(usage, model, batch=True) if usage else 0.0
                cost_total += cost
                for k in ("input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens"):
                    tok[k] += getattr(usage, k, 0) or 0
                if cl.apply_result(con, bid, model, pv, cid, res, usage, cost):
                    ok += 1
                    if cid in rows:
                        rows[cid].update({"label": res["label"], "rationale": res["rationale"],
                                          "contains_instructions": bool(res.get("contains_instructions")),
                                          "cost_usd": round(cost, 6)})
                        flagged += int(bool(res.get("contains_instructions")))
                else:
                    bad += 1
            else:
                bad += 1
                err = getattr(getattr(result.result, "error", None), "type", result.result.type)
                cl.apply_result(con, bid, model, pv, cid, None, None, 0.0, error=f"batch {result.result.type}: {err}")
        con.execute("UPDATE classifier_batches SET status = 'collected', n_succeeded = ?, n_errored = ?, cost_usd = ?, collected_at = ? "
                    "WHERE batch_id = ?", (ok, bad, round(cost_total, 4), state.now_iso(), bid))
        log(f"{bid}: {ok} labels, {bad} errored/unparseable, ${cost_total:.4f}; tokens {dict(tok)}; {flagged} window(s) flagged instruction-like text")
        meta = json.loads((OUT / "batch.json").read_text()) if (OUT / "batch.json").exists() else {}
        meta.update({"collected_at": state.now_iso(), "n_succeeded": ok, "n_errored": bad, "actual_usd": round(cost_total, 4),
                     "tokens": dict(tok), "flagged": flagged})
        (OUT / "batch.json").write_text(json.dumps(meta, indent=1))
    write_windows(list(rows.values()))
    return 0


# --- analyze -------------------------------------------------------------------------

def analyze(con):
    s = json.loads((REF / "sample.json").read_text())
    rows = read_windows()
    lab = [r for r in rows if r.get("label")]
    unl = [r for r in rows if r["words"] >= MIN_WORDS and not r.get("label")]
    if unl:
        log(f"WARNING: {len(unl)} windows still unlabelled")
    ref = reference()
    # 1. episodes exactly as the pipeline merges them (core bounds, gap 90 s, mixed labels -> about_both)
    for r in lab:
        r.setdefault("family_rates", {})
    eps = cl.merge_episodes(lab, gap=GAP, prefix="v2e")
    for e in eps:
        e["tags"] = cl.TAGS[e["label"]]
    eps_by = defaultdict(list)
    for e in eps:
        eps_by[e["video_id"]].append(e)
    half = 30.0
    res = {"at": state.now_iso(), "arm": "codebook v2, no keyword screen, every 120-s window of the whole-meeting WhisperX transcript",
           "window_s": WINDOW, "pad_s": PAD, "gap_s": GAP, "n_windows": len(rows), "n_sent": sum(1 for r in rows if r["words"] >= MIN_WORDS),
           "n_labelled": len(lab), "n_unlabelled": len(unl), "labels": dict(Counter(r["label"] for r in lab)),
           "instruction_flags": [{"id": r["id"], "start": rc.hms(r["start_s"]), "label": r["label"], "rationale": r.get("rationale")}
                                 for r in lab if r.get("contains_instructions")],
           "episodes": [{k: e[k] for k in ("episode_id", "video_id", "start_s", "end_s", "duration_s", "label", "tags", "windows")} for e in eps],
           "families": {}, "cost": {}}
    # 2. per family: recall (reference about-minutes inside v2 episodes) and precision (v2 episode-minutes the reference calls about)
    for fam, labels in FAMILIES.items():
        per = []
        for m in s["meetings"]:
            vid = m["video_id"]
            rr = [x for x in ref.get(vid, []) if x.get("label")]
            spans_any = [(e["start_s"], e["end_s"]) for e in eps_by[vid]]
            spans_tag = [(e["start_s"], e["end_s"]) for e in eps_by[vid] if e["tags"] in TAG_FAMILY[fam]]
            ab = [x for x in rr if x["label"] in labels]
            # precision: reference minutes inside family-tagged v2 episodes (>= 30 s), share whose reference label is in the family
            inside = [x for x in rr if rc.covered(spans_tag, x["start_s"], x["end_s"]) >= half]
            # window-level precision: v2 about-windows of the family whose reference minutes include a family label
            vw = [r for r in lab if r["video_id"] == vid and r["label"] in labels]
            per.append({"video_id": vid, "body": m["body"], "about_min": len(ab) * 1.0,
                        "located_min": sum(rc.covered(spans_any, x["start_s"], x["end_s"]) for x in ab) / 60,
                        "located_tag_min": sum(rc.covered(spans_tag, x["start_s"], x["end_s"]) for x in ab) / 60,
                        "found_w": sum(1 for x in ab if rc.covered(spans_any, x["start_s"], x["end_s"]) >= half), "n_w": len(ab),
                        "ep_min": len(inside) * 1.0, "ep_about_min": sum(1 for x in inside if x["label"] in labels) * 1.0,
                        "ep_anyabout_min": sum(1 for x in inside if x["label"] in cl.ABOUT) * 1.0,
                        "v2_w": len(vw), "v2_w_ref_about": sum(1 for r in vw if any(l in labels for l in r["ref_labels"])),
                        "v2_w_ref_anyabout": sum(1 for r in vw if any(l in cl.ABOUT for l in r["ref_labels"]))})
        tot = {k: sum(p[k] for p in per) for k in per[0] if k not in ("video_id", "body")}
        f = lambda n, d: round(n / d, 3) if d else None   # noqa: E731
        res["families"][fam] = {
            "about_min": tot["about_min"], "located_min": round(tot["located_min"], 1), "located_tag_min": round(tot["located_tag_min"], 1),
            "recall_min": f(tot["located_min"], tot["about_min"]), "recall_min_ci": rc.ratio_ci(per, "located_min", "about_min"),
            "recall_tag_min": f(tot["located_tag_min"], tot["about_min"]), "recall_tag_ci": rc.ratio_ci(per, "located_tag_min", "about_min"),
            "recall_window": f(tot["found_w"], tot["n_w"]), "recall_window_ci": rc.ratio_ci(per, "found_w", "n_w"),
            "episode_min": tot["ep_min"], "episode_about_min": tot["ep_about_min"],
            "precision_min": f(tot["ep_about_min"], tot["ep_min"]), "precision_min_ci": rc.ratio_ci(per, "ep_about_min", "ep_min"),
            "precision_min_anyabout": f(tot["ep_anyabout_min"], tot["ep_min"]),
            "v2_windows": tot["v2_w"], "precision_window": f(tot["v2_w_ref_about"], tot["v2_w"]),
            "precision_window_ci": rc.ratio_ci(per, "v2_w_ref_about", "v2_w"),
            "precision_window_anyabout": f(tot["v2_w_ref_anyabout"], tot["v2_w"]),
            "per_meeting": per}
    # 3. misses: reference about-minutes with < 30 s inside any v2 episode
    missed = []
    for vid, rr in ref.items():
        spans = [(e["start_s"], e["end_s"]) for e in eps_by[vid]]
        v2 = {r["i"]: r for r in lab if r["video_id"] == vid}
        for x in rr:
            if x.get("label") in cl.ABOUT and rc.covered(spans, x["start_s"], x["end_s"]) < half:
                w = v2.get(x["i"] // 2)
                missed.append({"video_id": vid, "start": rc.hms(x["start_s"]), "ref_label": x["label"], "v2_label": w["label"] if w else None,
                               "v2_rationale": (w or {}).get("rationale"), "text": " ".join((x.get("core_text") or x["text"]).split()[:28])})
    res["missed"] = missed
    res["miss_by_v2_label"] = dict(Counter(m["v2_label"] or "unsent" for m in missed))
    # 4. over-calls: v2 about-windows whose reference minutes hold no about_* label
    over = []
    for r in lab:
        if r["label"] in cl.ABOUT and not any(l in cl.ABOUT for l in r["ref_labels"]):
            kind = "mention_only" if "mention_only" in r["ref_labels"] else ("no" if r["ref_labels"] else "unlabelled")
            over.append({"id": r["id"], "video_id": r["video_id"], "start": rc.hms(r["start_s"]), "v2_label": r["label"], "ref_labels": r["ref_labels"],
                         "kind": kind, "rationale": r.get("rationale"), "text": " ".join(r["core_text"].split()[:28])})
    res["overcalls"] = over
    res["overcall_counts"] = dict(Counter(o["kind"] for o in over))
    # 5. cost: marginal Sonnet $ per transcript-hour of this arm
    hours = sum(m["audio_s"] for m in s["meetings"] if usable(m)) / 3600     # the 4 meetings actually sent
    res["excluded"] = [{"video_id": m["video_id"], "hours": m["hours"], "why": m.get("ingest_note")} for m in s["meetings"] if not usable(m)]
    toks = {k: int(con.execute(f"SELECT COALESCE(SUM({k}), 0) FROM classifier_calls WHERE video_id LIKE ?", (CALL_PREFIX + "%",)).fetchone()[0])
            for k in ("input_tokens", "cache_read_input_tokens", "cache_creation_input_tokens", "output_tokens")}
    actual = float(con.execute("SELECT COALESCE(SUM(cost_usd), 0) FROM classifier_calls WHERE video_id LIKE ?", (CALL_PREFIX + "%",)).fetchone()[0])
    est = json.loads((OUT / "estimate.json").read_text()) if (OUT / "estimate.json").exists() else {}
    res["cost"] = {"audio_hours": round(hours, 2), "est_usd": est.get("est_usd"), "worst_usd": est.get("worst_usd"), "actual_usd": round(actual, 4),
                   "usd_per_audio_hour": round(actual / hours, 4) if hours else None, "tokens": toks,
                   "tokens_per_audio_hour": {k: round(v / hours) for k, v in toks.items()} if hours else None,
                   "windows_per_audio_hour": round(len(lab) / hours, 1) if hours else None,
                   "corpus_2100h_usd": round(actual / hours * 2100, 0) if hours else None}
    (OUT / "results.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    # markdown
    L = ["| family | ref about-min | v2-episode-min | recall (min) | 95 % CI | recall (tag-matched) | precision (min) | 95 % CI | precision (any about) | v2 windows | window precision |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for fam, f in res["families"].items():
        if not f["about_min"]:
            L.append(f"| {fam} | 0 | {f['episode_min']:.0f} | - | - | - | {f['precision_min']} | - | - | {f['v2_windows']} | {f['precision_window']} |")
            continue
        ci, pci = f["recall_min_ci"], f["precision_min_ci"] or (None, None)
        L.append(f"| {fam} | {f['about_min']:.0f} | {f['episode_min']:.0f} | **{f['recall_min']:.3f}** | {ci[0]:.3f}-{ci[1]:.3f} | {f['recall_tag_min']:.3f} | "
                 f"{f['precision_min'] if f['precision_min'] is not None else '-'} | {pci[0]}-{pci[1]} | {f['precision_min_anyabout']} | {f['v2_windows']} | {f['precision_window']} |")
    L += ["", "| meeting | body | h | ref about-min | v2-located-min | recall | v2 about windows | v2 episodes (min) |", "|---|---|---|---|---|---|---|---|"]
    fa = {p["video_id"]: p for p in res["families"]["any"]["per_meeting"]}
    for m in s["meetings"]:
        p = fa[m["video_id"]]
        L.append(f"| {m['video_id']} | {m['body']} | {m['hours']:.2f} | {p['about_min']:.0f} | {p['located_min']:.1f} | "
                 f"{(p['located_min'] / p['about_min']) if p['about_min'] else float('nan'):.2f} | {p['v2_w']} | "
                 f"{len(eps_by[m['video_id']])} ({sum(e['duration_s'] for e in eps_by[m['video_id']]) / 60:.1f}) |")
    (OUT / "table.md").write_text("\n".join(L) + "\n")
    O = [f"# v2 full-ASR over-calls: {len(over)} v2 about-windows (120 s) whose reference minutes (codebook v1, 60 s) hold no about_* label",
         "", f"by reference verdict: {res['overcall_counts']}", ""]
    for kind in ("mention_only", "no", "unlabelled"):
        xs = [o for o in over if o["kind"] == kind]
        if not xs:
            continue
        O += [f"## reference said {kind} ({len(xs)})", ""]
        for o in xs:
            O.append(f"- **{o['video_id']} {o['start']}** v2 {o['v2_label']} (ref {'/'.join(o['ref_labels'])}) {o['rationale'] or ''} -- \"{o['text']}\"")
        O.append("")
    O += [f"# Misses: {len(missed)} reference about-minutes with < 30 s inside any v2 episode (v2 verdict on the covering window: {res['miss_by_v2_label']})", ""]
    for x in missed:
        O.append(f"- **{x['video_id']} {x['start']}** ref {x['ref_label']}, v2 {x['v2_label']} {x['v2_rationale'] or ''} -- \"{x['text']}\"")
    (OUT / "overcalls_and_misses.md").write_text("\n".join(O) + "\n")
    print("\n".join(L))
    print(f"\nlabels: {res['labels']}\nepisodes: {len(eps)} = {sum(e['duration_s'] for e in eps) / 60:.1f} min\n"
          f"over-calls: {res['overcall_counts']}\nmisses by v2 label: {res['miss_by_v2_label']}\ncost: {res['cost']}\n"
          f"instruction flags: {len(res['instruction_flags'])}")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["windows", "submit", "collect", "analyze", "v2"])
    ap.add_argument("--cap-usd", type=float, default=4.0)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--state", default=None)
    a = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    if a.cmd == "windows":
        build_windows()
        return 0
    if a.cmd == "v2":
        t, v = codebook_v2()
        print(v, prompt_version_v2(v), len(t), "chars")
        return 0
    con = state.connect(a.state)
    if a.cmd == "submit":
        return submit(con, a.cap_usd, a.dry_run)
    if a.cmd == "collect":
        return collect(con)
    return analyze(con)


if __name__ == "__main__":
    sys.exit(main())
