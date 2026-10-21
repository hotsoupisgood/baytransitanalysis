#!/usr/bin/env python3
"""
scripts/recall_check.py -- how much "about" speech does the captions-first
locate step miss? (plan/RECALL-CHECK.md, 2026-10-17.)

On meetings whose whole-meeting audio is already on disk (Richmond / San
Leandro, scripts/local_asr.py wavs under output/t2/asr/), transcribe the WHOLE
meeting on the GPU under the pinned corpus ASR configuration, label EVERY
60-s window with the pinned Sonnet codebook (no keyword screen), and compare
that exhaustive label set with what the pipeline located (segment.py screen at
0.5 hits/100w on the Mac-ASR track + the classifier -> `episodes`,
method='local_asr').

Nothing here touches output/whisper, corpus.sqlite or output/redacted: the GPU
batch is launched with WHISPER_DIR / BATCH_DIR pointed at
output/validation/recall/, the synthetic `videos` rows are kind='validation'
and in_frame=0, and the classifier rows are video_id 'recall:<id>' so no
per-meeting $ report ever sees them (spend.claude_total still does).

    sample   [--seed N]            choose 3 Richmond + 3 San Leandro meetings (~20 h)
    prepare                        wav -> 16 kHz mono opus; videos rows; batch row
                                   (then: gpu_batch.py approve <bid>; launch under nohup
                                   with WHISPER_DIR=... BATCH_DIR=... NOTIFY_BACKEND=none)
    tag-spend <bid>                note the spend / batches rows as validation
    harvest <bid>                  copy the pod's whole-meeting JSONs -> whisper/<id>.json
    windows                        every 60-s window of every transcript + the screen's verdicts
    submit   [--cap-usd 5.5]       Message Batch for all windows (classifier_calls, video_id recall:<id>)
    collect                        write the labels back
    analyze                        recall tables, bootstrap, missed windows, results.json
"""
import argparse, json, random, shutil, subprocess, sys, time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))
import state           # noqa: E402
import segment as sg   # noqa: E402
import classify as cl  # noqa: E402
import gpu_batch       # noqa: E402

OUT = ROOT / "output" / "validation" / "recall"
ASR_DIR = ROOT / "output" / "t2" / "asr"
BODIES = ("richmond", "sanleandro")
# The arm under measurement: which `episodes.method` the pipeline wrote on
# these meetings and which `videos.captions` kind the locate track is (the
# classifier instrument follows from it). scripts/recall_check_yt.py swaps
# these for the YouTube auto-caption arm ('classifier' / 'auto').
METHOD = "local_asr"
CAPTIONS_KIND = "asr_local"
WINDOW, PAD = 60.0, 30.0
PREFIX = "rc-"                      # synthetic videos rows
CALL_PREFIX = "recall:"             # classifier_calls video_id / id prefix
API_ID_PREFIX = "recall-"           # Batches API custom_id must be [a-zA-Z0-9_-]{1,64}; no ':'
NOTE = "validation: locate-recall check (plan/RECALL-CHECK.md)"
FAMILIES = {"density": {"about_density", "about_both"}, "transit": {"about_transit", "about_both"},
            "any": cl.ABOUT, "both_label": {"about_both"}}
TAG_FAMILY = {"density": {"density", "both"}, "transit": {"transit", "both"}, "any": {"density", "transit", "both"},
              "both_label": {"both"}}


def log(msg):
    print(f"{state.now_iso()} {msg}", flush=True)


def hms(s):
    return sg.hms(s)


def sample_path():
    return OUT / "sample.json"


def load_sample():
    return json.loads(sample_path().read_text())


# --- 1. sample ---------------------------------------------------------------------

def eligible(con):
    rows = []
    for r in con.execute("SELECT video_id, speaker, meeting_date, duration_s, status, captions_path, raw_meta FROM videos "
                         "WHERE kind = 'meeting' AND captions = 'asr_local' AND speaker IN (?, ?) ORDER BY speaker, meeting_date", BODIES):
        wav = ASR_DIR / f"{r['video_id']}.wav"
        if not wav.exists() or not r["captions_path"] or not Path(r["captions_path"]).exists():
            continue
        eps = con.execute("SELECT COUNT(*) n, COALESCE(SUM(end_s - start_s), 0) s, GROUP_CONCAT(DISTINCT tags) t FROM episodes "
                          "WHERE video_id = ? AND method = 'local_asr' AND status != 'mention'", (r["video_id"],)).fetchone()
        rows.append({"video_id": r["video_id"], "body": r["speaker"], "date": r["meeting_date"],
                     "hours": round((r["duration_s"] or 0) / 3600, 2), "status": r["status"], "wav": str(wav),
                     "vtt": r["captions_path"], "episodes": eps["n"], "episode_min": round(eps["s"] / 60, 1), "tags": eps["t"],
                     "asr": (json.loads(r["raw_meta"] or "{}").get("asr") or {}).get("model")})
    return rows


def sample(con, seed, per_body=3, lo_h=18.0, hi_h=23.0):
    pool = eligible(con)
    rng = random.Random(seed)
    for attempt in range(1, 10001):
        pick = []
        for b in BODIES:
            pick += rng.sample([r for r in pool if r["body"] == b], per_body)
        hours = sum(r["hours"] for r in pick)
        if lo_h <= hours <= hi_h and any(r["episodes"] == 0 for r in pick) and sum(1 for r in pick if r["episodes"] > 0) >= 4:
            break
    else:
        sys.exit("no draw satisfied the constraints")
    OUT.mkdir(parents=True, exist_ok=True)
    doc = {"seed": seed, "draws": attempt, "rule": f"{per_body} per body from the {len(pool)} asr_local meetings with the wav on disk; "
           f"total {lo_h:.0f}-{hi_h:.0f} h; >= 1 meeting with 0 located episodes; >= 4 with some",
           "hours": round(hours, 2), "meetings": pick, "at": state.now_iso()}
    sample_path().write_text(json.dumps(doc, indent=1))
    for r in pick:
        log(f"  {r['video_id']:20} {r['body']:10} {r['date']} {r['hours']:5.2f} h  {r['episodes']:2d} episodes {r['episode_min']:6.1f} min  {r['tags'] or '-'}")
    log(f"sample: {len(pick)} meetings, {hours:.2f} audio-h, seed {seed}, draw {attempt} -> {sample_path()}")
    return doc


# --- 2. prepare: opus + synthetic rows + batch row ------------------------------------

def transcode(wav, dst):
    cmd = ["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(wav), "-ac", "1", "-ar", "16000",
           "-c:a", "libopus", "-b:a", "16k", str(dst)]                   # pipeline/transcode.py, no -t cap
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg: {(r.stderr or '')[-200:]}")
    return time.time() - t0


def duration(path):
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1", str(path)],
                       capture_output=True, text=True, timeout=120)
    return float(r.stdout.strip())


def prepare(con):
    import vocab                                               # noqa: E402
    s = load_sample()
    audio = OUT / "audio"
    audio.mkdir(parents=True, exist_ok=True)
    now = state.now_iso()
    hours = 0.0
    for m in s["meetings"]:
        vid = m["video_id"]
        rid = PREFIX + vid
        dst = audio / f"{rid}.opus"
        if not dst.exists():
            wall = transcode(m["wav"], dst)
            log(f"  {vid}: opus {dst.stat().st_size / 1e6:.1f} MB in {wall:.0f} s")
        dur = duration(dst)
        hours += dur / 3600
        src = dict(con.execute("SELECT * FROM videos WHERE video_id = ?", (vid,)).fetchone())
        meta = {"validation": "recall_check", "source_video_id": vid, "meeting_id": src["meeting_id"], "body": src["speaker"],
                "initial_prompt": vocab.for_body(con, src["speaker"]), "diarize": False, "note": NOTE}
        if con.execute("SELECT 1 FROM videos WHERE video_id = ?", (rid,)).fetchone():
            con.execute("UPDATE videos SET opus_path = ?, transcribed_s = ?, duration_s = ?, raw_meta = ?, updated_at = ? WHERE video_id = ?",
                        (str(dst), round(dur, 1), round(dur, 1), json.dumps(meta), now, rid))
        else:
            con.execute("INSERT INTO videos (video_id, speaker, channel_url, month, title, duration_s, raw_meta, crawled_at, in_frame, "
                        "status, opus_path, transcribed_s, attempts, updated_at, kind, platform, source_url, meeting_id, meeting_date) "
                        "VALUES (?,?,?,?,?,?,?,?,0,'transcoded',?,?,0,?,'validation',?,?,?,?)",
                        (rid, src["speaker"], src["channel_url"] or src["source_url"] or "", (src["meeting_date"] or now)[:7],
                         f"[recall check] {src['title'] or vid}", round(dur, 1), json.dumps(meta), now, str(dst), round(dur, 1), now,
                         src["platform"], src["source_url"], src["meeting_id"], src["meeting_date"]))
        m["rid"], m["opus"], m["audio_s"] = rid, str(dst), round(dur, 1)
    c = gpu_batch.cfg()
    est = gpu_batch.estimate_usd(hours, c)
    import spend                                               # noqa: E402
    why = spend.runpod_refusal(con, est)
    if why:
        sys.exit(f"refusing: {why}")
    bid = s.get("batch_id")
    if not bid or con.execute("SELECT status FROM batches WHERE batch_id = ?", (bid,)).fetchone()["status"] not in ("awaiting_approval", "approved"):
        bid = gpu_batch.next_batch_id(con)
        con.execute("BEGIN IMMEDIATE")
        con.execute("INSERT INTO batches (batch_id, created_at, status, n_videos, audio_hours, est_usd, approval_requested_at, note) "
                    "VALUES (?,?,?,?,?,?,?,?)",
                    (bid, now, "awaiting_approval", len(s["meetings"]), round(hours, 3), est, now,
                     f"{NOTE}; whole-meeting audio, no diarization; est = ({hours:.2f} h / {c['rtfx']:.0f}x + {c['boot_min']:.0f} min) "
                     f"* ${c['usd_hr']:.2f}/h * {c['contingency']:.1f}"))
        con.executemany("INSERT INTO batch_videos (batch_id, video_id) VALUES (?, ?)", [(bid, m["rid"]) for m in s["meetings"]])
        con.execute("COMMIT")
        for m in s["meetings"]:
            state.transition(con, m["rid"], "transcoded", "queued", note=f"batch {bid} ({NOTE})")
    s.update({"batch_id": bid, "audio_hours": round(hours, 3), "gpu_est_usd": est, "prepared_at": now})
    sample_path().write_text(json.dumps(s, indent=1))
    log(f"batch {bid}: {len(s['meetings'])} whole meetings, {hours:.2f} audio-h, GPU est ${est:.2f} "
        f"(RTFX {c['rtfx']:.0f}x estimate, ${c['usd_hr']:.2f}/h, x{c['contingency']:.1f}). Status awaiting_approval.")
    log(f"next: .venv/bin/python pipeline/gpu_batch.py approve {bid} --via recall_check ; then launch under nohup with "
        f"WHISPER_DIR={OUT / 'whisper'} BATCH_DIR={OUT / 'batches'} NOTIFY_BACKEND=none")


def tag_spend(con, bid):
    b = dict(con.execute("SELECT * FROM batches WHERE batch_id = ?", (bid,)).fetchone())
    n = 0
    if b.get("pod_id"):
        n = con.execute("UPDATE spend SET note = ? WHERE pod_id = ? AND (note IS NULL OR note NOT LIKE 'validation:%')",
                        (f"{NOTE}; batch {bid}; " + "teardown", b["pod_id"])).rowcount
    if not (b.get("note") or "").startswith("validation:"):
        con.execute("UPDATE batches SET note = ? WHERE batch_id = ?", (f"{NOTE}; {b.get('note') or ''}"[:1000], bid))
    rows = [dict(r) for r in con.execute("SELECT * FROM spend WHERE pod_id = ?", (b.get("pod_id") or "",))]
    log(f"{bid}: status {b['status']}, pod {b.get('pod_id')}, gpu_minutes {b.get('gpu_minutes')}, actual ${b.get('actual_usd')}, "
        f"{n} spend row(s) tagged: " + "; ".join(f"${r['usd']:.4f} {r['gpu_hours'] * 60:.1f} min" for r in rows))


# --- 3. harvest the whole-meeting transcripts -------------------------------------------

def harvest(con, bid):
    s = load_sample()
    wdir = OUT / "whisper"
    wdir.mkdir(parents=True, exist_ok=True)
    staging = OUT / "batches" / bid / "out"
    got = []
    for m in s["meetings"]:
        rid, vid = m["rid"], m["video_id"]
        src = None
        for cand in (wdir / f"{rid}.json", staging / f"{rid}.json"):
            if cand.exists():
                src = cand
                break
        dst = wdir / f"{vid}.json"
        if src is None:
            fail = staging / "FAILED" / f"{rid}.txt"
            log(f"  {vid}: NO transcript" + (f" -- pod FAILED: {fail.read_text()[:200].strip()}" if fail.exists() else ""))
            continue
        if src != dst:
            shutil.copy2(src, dst)
        doc = json.loads(dst.read_text())
        row = con.execute("SELECT status, last_error FROM videos WHERE video_id = ?", (rid,)).fetchone()
        words = sum(len(x["text"].split()) for x in doc["segments"])
        m.update({"transcript": str(dst), "words": words, "segments": len(doc["segments"]), "gpu_seconds": doc.get("gpu_seconds"),
                  "model": doc.get("model"), "ingest_status": row["status"], "ingest_note": (row["last_error"] or "")[:300]})
        got.append(vid)
        log(f"  {vid}: {len(doc['segments'])} segments, {words:,} words, {doc.get('gpu_seconds')} GPU-s, model {doc.get('model')}, "
            f"row {row['status']}" + (f" ({row['last_error'][:120]})" if row["last_error"] else ""))
    b = dict(con.execute("SELECT * FROM batches WHERE batch_id = ?", (bid,)).fetchone())
    s.update({"gpu_actual_usd": b.get("actual_usd"), "gpu_minutes": b.get("gpu_minutes"), "batch_note": b.get("note"), "harvested": got})
    sample_path().write_text(json.dumps(s, indent=1))
    log(f"harvested {len(got)}/{len(s['meetings'])}; batch {bid} {b['status']}, {b.get('gpu_minutes')} GPU-min, actual ${b.get('actual_usd')} vs est ${b.get('est_usd')}")


# --- 4. windows ---------------------------------------------------------------------------

def windows_path():
    return OUT / "windows.jsonl"


def read_windows():
    return [json.loads(l) for l in windows_path().read_text().splitlines() if l.strip()]


def write_windows(rows):
    windows_path().write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


def covered(spans, a, b):
    """Seconds of [a, b) inside the union of spans [(s, e)]."""
    tot = 0.0
    for s, e in sorted(spans):
        lo, hi = max(a, s), min(b, e)
        if hi > lo:
            tot += hi - lo
    return min(tot, b - a)


def pipeline_verdicts(con, vid, segs_asr, keywords, p=cl.DEFAULTS):
    """What the pipeline did on this meeting's Mac-ASR track: the screen's
    candidate spans (segment.candidate_windows at 0.5 / 100 w, 3-window
    smoothing) and the classifier's label per span (classifier_calls, the
    same instrument, bounds-matched like classify.cached_calls)."""
    cands = cl.candidates(vid, segs_asr, keywords, p)
    model = state.setting("SCORING_MODEL", cl.LEGACY_MODEL)
    pv = cl.prompt_version_for(model, cl.DEFAULT_LANG, CAPTIONS_KIND)
    have = cl.cached_calls(con, vid, model, pv, cands)
    labels = {}
    for c in cands:
        r = have.get(c["id"])
        labels[c["id"]] = r["label"] if r else None
    # any labelled row for the video that covers a span (older candidate sets)
    extra = [dict(r) for r in con.execute("SELECT start_s, end_s, label FROM classifier_calls WHERE video_id = ? AND label IS NOT NULL "
                                          "ORDER BY at", (vid,))]
    return cands, labels, extra


def build_windows(con):
    s = load_sample()
    keywords = cl.keywords_for("en")
    rows = []
    for m in s["meetings"]:
        if not m.get("transcript"):
            continue
        vid = m["video_id"]
        doc = json.loads(Path(m["transcript"]).read_text())
        segs = [{"start": float(x["start"]), "end": float(x["end"]), "text": x["text"]} for x in doc["segments"]]
        end_of_talk = max((x["end"] for x in segs), default=0)
        segs_asr = sg.read_segments(m["vtt"])
        cands, cand_labels, extra = pipeline_verdicts(con, vid, segs_asr, keywords)
        cand_spans = [(c["start_s"], c["end_s"]) for c in cands]
        # screen on the GPU text itself, to separate "the words were not there" from "the screen does not fire"
        gpu_cands = sg.candidate_windows(segs, keywords, WINDOW, cl.DEFAULTS["threshold"], cl.DEFAULTS["max_run"])
        gpu_spans = [(c["start_s"], c["end_s"]) for c in gpu_cands]
        eps = [dict(r) for r in con.execute("SELECT episode_id, start_s, end_s, tags, status, label FROM episodes WHERE video_id = ? "
                                            "AND method = ? AND status != 'mention' ORDER BY start_s", (vid, METHOD))]
        n = int(end_of_talk // WINDOW) + 1
        for i in range(n):
            a, b = i * WINDOW, (i + 1) * WINDOW
            pa, pb = max(0.0, a - PAD), min(end_of_talk, b + PAD)
            core = cl.window_text(segs, a, b)
            text = cl.window_text(segs, pa, pb)
            words = len(text.split())
            c = sg.hits(text, keywords)
            fh = sg.family_hits(text, keywords)
            # the pipeline's verdict on this minute
            scr = covered(cand_spans, a, b)
            lab = None
            for cnd in cands:
                if cnd["start_s"] <= a < cnd["end_s"] or (cnd["start_s"] < b and cnd["end_s"] > a and covered([(cnd["start_s"], cnd["end_s"])], a, b) >= WINDOW / 2):
                    lab = cand_labels.get(cnd["id"])
                    break
            if lab is None and scr > 0:
                for r in extra:
                    if r["start_s"] < b and r["end_s"] > a:
                        lab = r["label"]
            ep_cov = covered([(e["start_s"], e["end_s"]) for e in eps], a, b)
            ep_tags = sorted({e["tags"] for e in eps if e["start_s"] < b and e["end_s"] > a})
            rows.append({"id": f"{CALL_PREFIX}{vid}_w{i:04d}", "video_id": vid, "body": m["body"], "i": i, "start_s": a, "end_s": b,
                         "pad_start_s": pa, "pad_end_s": pb, "words": words, "core_words": len(core.split()),
                         "hits_per_100w": round(sg.density_per_100w(words, sum(c.values())), 2),
                         "family_rates": {f: round(sg.density_per_100w(words, k), 2) for f, k in sorted(fh.items())},
                         "top_keywords": [t for t, _ in c.most_common(5)],
                         "screen_s": round(scr, 1), "screen_gpu_s": round(covered(gpu_spans, a, b), 1), "pipeline_label": lab,
                         "episode_s": round(ep_cov, 1), "episode_tags": ep_tags,
                         "text": text, "core_text": core, "label": None})
        log(f"  {vid}: {n} windows, {sum(1 for r in rows if r['video_id'] == vid and r['words'] >= 10)} with >= 10 words; "
            f"screen marked {sum(1 for r in rows if r['video_id'] == vid and r['screen_s'] >= WINDOW / 2)} min on the ASR track, "
            f"{sum(1 for r in rows if r['video_id'] == vid and r['screen_gpu_s'] >= WINDOW / 2)} on the GPU text; "
            f"{len(eps)} located episodes = {sum(e['end_s'] - e['start_s'] for e in eps) / 60:.1f} min")
    write_windows(rows)
    model = state.setting("SCORING_MODEL", cl.LEGACY_MODEL)
    send = [r for r in rows if r["words"] >= 10]
    tokens = sum(cl.caption_tokens_for(r["words"]) for r in send)
    est = cl.estimate_usd(tokens, len(send), model, batch=True)
    worst = sum(cl.worst_case_usd(r["text"], model, batch=True) for r in send)
    log(f"{len(rows)} windows ({len(send)} with >= 10 words, {sum(r['words'] for r in send):,} padded words, ~{tokens:,} tokens); "
        f"model {model}; Claude estimate ${est:.2f} (worst case ${worst:.2f}) at the batch rate -> {windows_path()}")
    s.update({"n_windows": len(rows), "n_to_label": len(send), "padded_words": sum(r["words"] for r in send),
              "claude_est_usd": round(est, 2), "claude_worst_usd": round(worst, 2)})
    sample_path().write_text(json.dumps(s, indent=1))


# --- 5. submit / collect (Message Batches API, classify.py's request shape) --------------------

def submit(con, cap_usd, dry_run=False):
    import spend                                               # noqa: E402
    rows = read_windows()
    model = state.setting("SCORING_MODEL", cl.LEGACY_MODEL)
    pv = cl.prompt_version_for(model)                          # the bare T0 hash = the local_asr route's own instrument
    done = {r["id"] for r in con.execute("SELECT id FROM classifier_calls WHERE video_id LIKE ? AND label IS NOT NULL", (CALL_PREFIX + "%",))}
    send = [r for r in rows if r["words"] >= 10 and r["id"] not in done]
    tokens = sum(cl.caption_tokens_for(r["words"]) for r in send)
    est = cl.estimate_usd(tokens, len(send), model, batch=True)
    worst = sum(cl.worst_case_usd(r["text"], model, batch=True) for r in send)
    log(f"submit: {len(send)} windows ({len(done)} already labelled), model {model}, prompt_version {pv}, "
        f"estimate ${est:.2f}, worst ${worst:.2f}, cap ${cap_usd}")
    if est > cap_usd:
        log(f"REFUSED: estimate ${est:.2f} exceeds the ${cap_usd} ceiling for this check")
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
    provisional = f"pending-recall-{now}"
    con.execute("INSERT INTO classifier_batches (batch_id, created_at, status, model, prompt_version, n_requests, est_usd, note) "
                "VALUES (?,?,?,?,?,?,?,?)", (provisional, now, "recall_check", model, pv, len(send), round(est, 4), NOTE))
    reqs = []
    for r in send:
        c = {"id": r["id"], "start_s": r["start_s"], "end_s": r["end_s"], "pad_start_s": r["pad_start_s"], "pad_end_s": r["pad_end_s"],
             "words": r["words"], "hits_per_100w": r["hits_per_100w"], "family_rates": r["family_rates"], "top_keywords": r["top_keywords"]}
        rec = cl.call_record(c, CALL_PREFIX + r["video_id"], model, pv, batch_id=provisional)
        cl.insert_call(con, cl.call_row(rec), replace=True)
        reqs.append({"custom_id": API_ID_PREFIX + r["id"][len(CALL_PREFIX):],
                     "params": cl.request_params(model, r["text"], cl.DEFAULT_LANG)})
    try:
        mb = client.messages.batches.create(requests=reqs)
    except Exception as e:                                     # noqa: BLE001 -- nothing billed; undo
        con.execute("DELETE FROM classifier_calls WHERE batch_id = ? AND label IS NULL", (provisional,))
        con.execute("DELETE FROM classifier_batches WHERE batch_id = ?", (provisional,))
        log(f"batch create failed: {type(e).__name__}: {str(e)[:200]}; provisional rows removed")
        return 1
    con.execute("UPDATE classifier_batches SET batch_id = ?, note = ? WHERE batch_id = ?",
                (mb.id, f"{NOTE}; processing_status {mb.processing_status}", provisional))
    con.execute("UPDATE classifier_calls SET batch_id = ? WHERE batch_id = ?", (mb.id, provisional))
    s = load_sample()
    s.setdefault("claude_batches", []).append(mb.id)
    sample_path().write_text(json.dumps(s, indent=1))
    log(f"submitted {mb.id}: {len(reqs)} requests, status {mb.processing_status}")
    return 0


def collect(con):
    import anthropic
    client = anthropic.Anthropic()
    batches = [dict(r) for r in con.execute("SELECT * FROM classifier_batches WHERE status = 'recall_check' ORDER BY created_at")]
    if not batches:
        log("no recall_check batches pending")
        return 0
    rows = {r["id"]: r for r in read_windows()}
    for b in batches:
        bid, model, pv = b["batch_id"], b["model"], b["prompt_version"]
        mb = client.messages.batches.retrieve(bid)
        if mb.processing_status != "ended":
            rc = mb.request_counts
            log(f"{bid}: {mb.processing_status} ({rc.processing} processing, {rc.succeeded} succeeded, {rc.errored} errored)")
            continue
        ok = bad = flagged = 0
        cost_total = 0.0
        for result in client.messages.batches.results(bid):
            cid = result.custom_id
            if cid.startswith(API_ID_PREFIX):
                cid = CALL_PREFIX + cid[len(API_ID_PREFIX):]
            if result.result.type == "succeeded":
                msg = result.result.message
                res, usage = cl.parse_reply(msg), msg.usage
                cost = cl.usd(usage, model, batch=True) if usage else 0.0
                cost_total += cost
                if cl.apply_result(con, bid, model, pv, cid, res, usage, cost):
                    ok += 1
                    if cid in rows:
                        rows[cid]["label"] = res["label"]
                        rows[cid]["rationale"] = res["rationale"]
                        rows[cid]["contains_instructions"] = bool(res.get("contains_instructions"))
                        flagged += int(bool(res.get("contains_instructions")))
                else:
                    bad += 1
            else:
                bad += 1
                err = getattr(getattr(result.result, "error", None), "type", result.result.type)
                cl.apply_result(con, bid, model, pv, cid, None, None, 0.0, error=f"batch {result.result.type}: {err}")
        con.execute("UPDATE classifier_batches SET status = 'collected', n_succeeded = ?, n_errored = ?, cost_usd = ?, collected_at = ? "
                    "WHERE batch_id = ?", (ok, bad, round(cost_total, 4), state.now_iso(), bid))
        log(f"{bid}: {ok} labels, {bad} errored/unparseable, ${cost_total:.4f}; {flagged} window(s) flagged instruction-like text")
    write_windows(list(rows.values()))
    s = load_sample()
    s["claude_actual_usd"] = round(float(con.execute("SELECT COALESCE(SUM(cost_usd), 0) FROM classifier_calls WHERE video_id LIKE ?",
                                                    (CALL_PREFIX + "%",)).fetchone()[0]), 4)
    sample_path().write_text(json.dumps(s, indent=1))
    return 0


# --- 6. analyze ----------------------------------------------------------------------------

def ratio_ci(per_meeting, num_key, den_key, n_boot=4000, seed=7):
    """Percentile 95 % CI of sum(num)/sum(den) under a meeting-cluster bootstrap."""
    rng = random.Random(seed)
    ms = list(per_meeting)
    vals = []
    for _ in range(n_boot):
        pick = [rng.choice(ms) for _ in ms]
        d = sum(p[den_key] for p in pick)
        if d:
            vals.append(sum(p[num_key] for p in pick) / d)
    if not vals:
        return None, None
    vals.sort()
    return round(vals[int(0.025 * len(vals))], 3), round(vals[min(len(vals) - 1, int(0.975 * len(vals)))], 3)


def analyze(con):
    s = load_sample()
    rows = [r for r in read_windows() if r.get("label")]
    unl = [r for r in read_windows() if r["words"] >= 10 and not r.get("label")]
    if unl:
        log(f"WARNING: {len(unl)} windows still unlabelled")
    half = WINDOW / 2
    res = {"at": state.now_iso(), "sample": s, "families": {}, "meetings": {}, "screen": {}, "labels": dict(Counter(r["label"] for r in rows)),
           "n_windows_labelled": len(rows), "n_unlabelled": len(unl),
           "instruction_flags": [{"id": r["id"], "start": hms(r["start_s"]), "label": r["label"], "rationale": r.get("rationale")}
                                 for r in rows if r.get("contains_instructions")]}
    missed = []
    for fam, labels in FAMILIES.items():
        per = []
        for m in s["meetings"]:
            vid = m["video_id"]
            ab = [r for r in rows if r["video_id"] == vid and r["label"] in labels]
            # minutes: located (any tag) and located with a family-matching tag; window-level found = >= half covered
            tagged = [r for r in ab if any(t in TAG_FAMILY[fam] for t in r["episode_tags"])]
            per.append({"video_id": vid, "body": m["body"], "about_min": len(ab) * WINDOW / 60,
                        "located_min": sum(r["episode_s"] for r in ab) / 60,
                        "located_tag_min": sum(r["episode_s"] for r in tagged) / 60,
                        "found_w": sum(1 for r in ab if r["episode_s"] >= half), "n_w": len(ab),
                        "screened_w": sum(1 for r in ab if r["screen_s"] >= half),
                        "screened_gpu_w": sum(1 for r in ab if r["screen_gpu_s"] >= half),
                        "episodes_total_min": sum(float(e[0]) for e in con.execute(
                            "SELECT (end_s - start_s) / 60.0 FROM episodes WHERE video_id = ? AND method = ? AND status != 'mention'", (vid, METHOD)))})
        tot = {k: sum(p[k] for p in per) for k in ("about_min", "located_min", "located_tag_min", "found_w", "n_w", "screened_w", "screened_gpu_w")}
        rec = (tot["located_min"] / tot["about_min"]) if tot["about_min"] else None
        res["families"][fam] = {
            "about_min": tot["about_min"], "located_min": round(tot["located_min"], 1), "located_tag_min": round(tot["located_tag_min"], 1),
            "recall_min": round(rec, 3) if rec is not None else None, "recall_min_ci": ratio_ci(per, "located_min", "about_min"),
            "recall_tag_min": round(tot["located_tag_min"] / tot["about_min"], 3) if tot["about_min"] else None,
            "recall_tag_ci": ratio_ci(per, "located_tag_min", "about_min"),
            "recall_window": round(tot["found_w"] / tot["n_w"], 3) if tot["n_w"] else None, "recall_window_ci": ratio_ci(per, "found_w", "n_w"),
            "screen_recall_window": round(tot["screened_w"] / tot["n_w"], 3) if tot["n_w"] else None,
            "screen_recall_ci": ratio_ci(per, "screened_w", "n_w"),
            "screen_recall_gpu_text": round(tot["screened_gpu_w"] / tot["n_w"], 3) if tot["n_w"] else None,
            "per_meeting": per}
    # screen precision (all windows): marked on the ASR track -> is the minute about anything?
    marked = [r for r in rows if r["screen_s"] >= half]
    marked_gpu = [r for r in rows if r["screen_gpu_s"] >= half]
    res["screen"] = {"marked_w": len(marked), "marked_about_w": sum(1 for r in marked if r["label"] in cl.ABOUT),
                     "precision": round(sum(1 for r in marked if r["label"] in cl.ABOUT) / len(marked), 3) if marked else None,
                     "marked_gpu_w": len(marked_gpu), "precision_gpu_text": round(sum(1 for r in marked_gpu if r["label"] in cl.ABOUT) / len(marked_gpu), 3) if marked_gpu else None,
                     "labelled_w": len(rows), "about_w": sum(1 for r in rows if r["label"] in cl.ABOUT)}
    # the misses, decomposed: screen (never a candidate) vs classifier (candidate, labelled not-about) vs other
    for r in rows:
        if r["label"] in cl.ABOUT and r["episode_s"] < half:
            if r["screen_s"] < half:
                why = "screen"
            elif r["pipeline_label"] in cl.ABOUT or r["pipeline_label"] is None:
                why = "other" if r["pipeline_label"] else "screen-only/unlabelled"
            else:
                why = "classifier"
            missed.append({"video_id": r["video_id"], "body": r["body"], "start": hms(r["start_s"]), "start_s": r["start_s"], "label": r["label"],
                           "why": why, "pipeline_label": r["pipeline_label"], "screen_s": r["screen_s"], "hits_per_100w": r["hits_per_100w"],
                           "screen_gpu_s": r["screen_gpu_s"], "rationale": r.get("rationale"),
                           "text": " ".join((r["core_text"] or r["text"]).split()[:28])})
    res["missed"] = missed
    res["miss_share"] = dict(Counter(m["why"] for m in missed))
    # found / missed per label
    res["by_label"] = {lab: {"n": sum(1 for r in rows if r["label"] == lab), "found": sum(1 for r in rows if r["label"] == lab and r["episode_s"] >= half)}
                       for lab in cl.ABOUT}
    (OUT / "results.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    # markdown pieces
    lines = ["| family | about-min (exhaustive) | located-min (any tag) | recall (min) | 95 % CI | recall (tag-matched) | window recall | screen-alone recall | screen recall on GPU text |", "|---|---|---|---|---|---|---|---|---|"]
    for fam, f in res["families"].items():
        ci = f["recall_min_ci"]
        lines.append(f"| {fam} | {f['about_min']:.0f} | {f['located_min']:.1f} | {f['recall_min']:.3f} | {ci[0]:.3f}-{ci[1]:.3f} | {f['recall_tag_min']:.3f} | "
                     f"{f['recall_window']:.3f} | {f['screen_recall_window']:.3f} | {f['screen_recall_gpu_text']:.3f} |" if f["about_min"] else f"| {fam} | 0 | - | - | - | - | - | - | - |")
    lines += ["", "| meeting | body | h | about-min any | located-min | recall | about density | about transit | about both | episodes located (min) |", "|---|---|---|---|---|---|---|---|---|---|"]
    fa = {p["video_id"]: p for p in res["families"]["any"]["per_meeting"]}
    for m in s["meetings"]:
        p = fa[m["video_id"]]
        cnt = Counter(r["label"] for r in rows if r["video_id"] == m["video_id"])
        lines.append(f"| {m['video_id']} | {m['body']} | {m['hours']:.2f} | {p['about_min']:.0f} | {p['located_min']:.1f} | "
                     f"{(p['located_min'] / p['about_min']) if p['about_min'] else float('nan'):.2f} | {cnt['about_density']} | {cnt['about_transit']} | {cnt['about_both']} | {m['episodes']} ({p['episodes_total_min']:.1f}) |")
    (OUT / "table.md").write_text("\n".join(lines) + "\n")
    track = "YouTube auto-caption locate track" if CAPTIONS_KIND == "auto" else "Mac-ASR locate track"
    ml = [f"# Missed about-windows ({len(missed)} of {res['screen']['about_w']} about-minutes; exhaustive labels on GPU WhisperX text, "
          f"pipeline located on the {track})", "",
          f"why = screen: the keyword screen never marked this minute on the {track}; classifier: it was a candidate and Sonnet "
          "labelled it mention_only / no; screen-only/unlabelled: a candidate under this keyword list that the classifier has not read; "
          "other: candidate labelled about but outside the merged episode bounds.", ""]
    bodies = list(dict.fromkeys(m["body"] for m in s["meetings"]))
    for body in bodies:
        for vid in [m["video_id"] for m in s["meetings"] if m["body"] == body]:
            mm = [x for x in missed if x["video_id"] == vid]
            ml += [f"## {vid} ({body}) -- {len(mm)} missed", ""]
            for x in mm:
                ml.append(f"- **{x['start']}** {x['label']} [{x['why']}; pipeline {x['pipeline_label'] or '-'}; {x['hits_per_100w']} hits/100w] "
                          f"{x['rationale'] or ''} -- \"{x['text']}\"")
            ml.append("")
    (OUT / "missed_windows.md").write_text("\n".join(ml))
    print("\n".join(lines))
    print(f"\nlabels: {res['labels']}\nmisses by cause: {res['miss_share']}\nscreen: {res['screen']}\nby_label: {res['by_label']}\n"
          f"instruction flags: {len(res['instruction_flags'])}")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["sample", "prepare", "tag-spend", "harvest", "windows", "submit", "collect", "analyze"])
    ap.add_argument("id", nargs="?")
    ap.add_argument("--seed", type=int, default=20261017)
    ap.add_argument("--cap-usd", type=float, default=5.5)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--state", default=None)
    a = ap.parse_args(argv)
    con = state.connect(a.state)
    OUT.mkdir(parents=True, exist_ok=True)
    if a.cmd == "sample":
        sample(con, a.seed)
    elif a.cmd == "prepare":
        prepare(con)
    elif a.cmd == "tag-spend":
        tag_spend(con, a.id)
    elif a.cmd == "harvest":
        harvest(con, a.id)
    elif a.cmd == "windows":
        build_windows(con)
    elif a.cmd == "submit":
        return submit(con, a.cap_usd, a.dry_run)
    elif a.cmd == "collect":
        return collect(con)
    elif a.cmd == "analyze":
        return analyze(con)
    return 0


if __name__ == "__main__":
    sys.exit(main())
