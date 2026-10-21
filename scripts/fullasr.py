#!/usr/bin/env python3
"""
scripts/fullasr.py -- full-ASR locate with codebook v2.1 for the Granicus and
Swagit bodies (plan/FULLASR-GRANICUS.md; approved by Puck 2026-10-18 10:13 PDT).

The whole meeting is transcribed ONCE on the GPU without diarization (the b35
validation configuration: WhisperX large-v3, alignment, the body's vocabulary,
`diarize: False`, outputs under output/fullasr/whisper/, never output/whisper/),
then EVERY 120-s window of that transcript is read by Sonnet under codebook
v2.1 with no keyword screen (DECISIONS-PROPOSED-13 option 2 + PROPOSED-14),
adjacent about_* windows merge into `episodes` rows with method='fullasr' (a
new ROUTE value, SCHEMA.md), and the spans no transcribed v1 clip covers are
cut offline from the whole-meeting audio on disk and registered as clips for a
normal diarized episode batch.

Nothing here touches .env, plan/frame.csv, plan/DECISIONS.md, any v1 row, or
YouTube. Audio comes from the Granicus / Swagit archive hosts through the
adapters' politeness rules (>= 5 s between requests, one ffmpeg, stop on
403/429), or is reused from disk (output/t2/asr/<id>.wav, the b35 transcripts).

    scope                       the meeting table per body -> output/fullasr/scope.json
    fetch [--bodies] [--max-hours H] [--max N]   whole-meeting audio -> output/fullasr/audio/<id>.opus (64k master) + opus/<id>.opus (16k, GPU)
    encode                      wav / flac on disk -> output/fullasr/opus/<id>.opus (16 kHz mono 16 kbps)
    reuse                       the four usable b35 whole-meeting transcripts -> output/fullasr/whisper/
    prepare --hours H [--bodies]   register 'fa-<id>' validation rows + ONE batch row (awaiting_approval)
    harvest <bid>               fa-<id>.json -> whisper/<id>.json; est vs actual recorded
    rescue <bid> [--dry-run]    QA-failed rows whose ONLY failure is a dead-air filler run ("you" x n over a recess): re-gate with the run collapsed, load if it passes
    windows                     every 120-s window (+- 30 s pad) of every new transcript -> windows/<id>.jsonl
    submit [--cap-usd]          one Message Batch, codebook v2.1, prompt_version v2.1-.../mr120, note fullasr-v2.1
    collect                     labels back; meetings fully labelled -> episodes (method fullasr) -> cut
    derive [--ids ...]          episodes + cut for meetings whose windows are all labelled (collect does this too)
    propose                     >= 8 h of clips ready and no pod live -> gpu_batch.propose --speaker <bodies>
    report                      tables for plan/FULLASR-GRANICUS.md
"""
import argparse, json, re, shutil, subprocess, sys, time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))
sys.path.insert(0, str(ROOT / "scripts"))
import state                      # noqa: E402
import segment as sg              # noqa: E402
import classify as cl             # noqa: E402
import granicus                   # noqa: E402
import episodes as epm            # noqa: E402
import transcode                  # noqa: E402
import gpu_batch                  # noqa: E402
import spend                      # noqa: E402
import vocab                      # noqa: E402
from downloader import ffprobe_duration   # noqa: E402

OUT = ROOT / "output" / "fullasr"
AUDIO, OPUS, WHISPER = OUT / "audio", OUT / "opus", OUT / "whisper"
BATCHES, WINDOWS, CLIPS, LOGS = OUT / "batches", OUT / "windows", OUT / "clips", OUT / "logs"
SCOPE = OUT / "scope.json"
GPU_LOG = OUT / "gpu_batches.json"
LOCAL_ASR = ROOT / "output" / "t2" / "asr"
RECALL = ROOT / "output" / "validation" / "recall"
SUBS = ROOT / "output" / "subs"

ROW_PREFIX = "fa-"                      # videos row of the whole-meeting GPU job (kind 'validation')
CALL_PREFIX = "fullasr:"                # classifier_calls.video_id / id prefix: never seen by cached_calls
API_ID_PREFIX = "fullasr-"              # Message Batches custom_id (no ':' allowed)
BATCH_STATUS = "fullasr_v21"            # classifier_batches.status: classify.collect ignores it
BATCH_NOTE = "fullasr-v2.1"
CODEBOOK = "v2.1"
MODEL = cl.LEGACY_MODEL
WINDOW, PAD, GAP, MIN_WORDS = 120.0, cl.DEFAULTS["pad"], cl.DEFAULTS["gap"], 10
METHOD = "fullasr"                      # episodes.method: the ROUTE (whole-meeting GPU ASR, no screen)
CUT_SOURCE = "fullasr_audio"            # raw_meta.cut_source on a clip cut from the whole-meeting file here
GPU_NOTE = "fullasr: whole-meeting ASR, no diarization (plan/FULLASR-GRANICUS.md)"
MIN_MEETING_S = 600.0                   # a 'meeting' shorter than this is a stub recording: listed, not transcribed
EXCLUDE = {"gr-richmond-6033": "b35 GPU transcript failed QA: decoder loop, coverage 5 % (RECALL-CHECK)",
           "gr-richmond-6038": "b35 GPU transcript failed QA: decoder loop, coverage 5 % (RECALL-CHECK)"}
HLS_BLOCKED = epm.HLS_BLOCKED
# bodies whose audio is already on disk come first in `prepare` (no download wait);
# sanfrancisco / sfmta were added to the scope after the ten original bodies and
# sit last (no audio on disk, nothing read under v1 either)
BODY_ORDER = ["richmond", "sanleandro", "hayward", "santaclara", "napa", "oakland", "fremont", "sausalito", "samtrans", "petaluma",
              "sanfrancisco", "sfmta"]


def log(msg):
    print(f"{state.now_iso()} {msg}", flush=True)


def hms(s):
    return sg.hms(s)


def pv():
    return cl.prompt_version_for(MODEL, cl.DEFAULT_LANG, None, codebook=CODEBOOK, max_run=WINDOW)


def jload(p, default=None):
    return json.loads(Path(p).read_text()) if Path(p).exists() else default


def jdump(p, obj):
    Path(p).write_text(json.dumps(obj, indent=1, ensure_ascii=False))


# --- 1. scope ------------------------------------------------------------------------------

def meetings(con, bodies=None):
    """In-scope meetings: the bodies of BODY_ORDER. The ten Granicus/Swagit
    bodies Puck approved on WhatsApp 2026-10-18 10:13 PDT, plus `sanfrancisco`
    and `sfmta` (Granicus rows that landed 2026-10-18 20:57 UTC from a sibling
    crawl and were added to this run's scope the same evening)."""
    bodies = list(bodies) if bodies else list(BODY_ORDER)
    where = f"kind = 'meeting' AND platform != 'youtube' AND speaker IN ({','.join('?' * len(bodies))})"
    params = bodies
    return [dict(r) for r in con.execute(f"SELECT * FROM videos WHERE {where} ORDER BY speaker, meeting_date, video_id", params)]


def meta_of(row):
    try:
        return json.loads(row.get("raw_meta") or "{}")
    except (ValueError, TypeError):
        return {}


def audio_on_disk(vid):
    """(kind, path) of the whole-meeting audio already on this Mac, or (None, None)."""
    for kind, p in (("wav", LOCAL_ASR / f"{vid}.wav"), ("flac", AUDIO / f"{vid}.flac"), ("opus", AUDIO / f"{vid}.opus")):
        if p.exists() and p.stat().st_size > 0:
            return kind, p
    return None, None


def audio_url(row):
    """(kind, url) of the archive source to pull: progressive MP4 first (one
    HTTP stream, ~85x real time measured 2026-10-18), else HLS (~6x), else the
    ASX.php -> MP4 path (Richmond / San Leandro players)."""
    m = meta_of(row)
    # fremont / samtrans listings carry the STRING 'None' for mp4 (crawler str()); it must fall through to HLS
    if m.get("mp4") and str(m["mp4"]).startswith("http"):
        return "mp4", m["mp4"]
    hls = m["m3u8"] if m.get("m3u8") and str(m["m3u8"]).startswith("http") else None
    if row.get("platform") == "granicus":
        # 2026-10-18 probe: fremont / samtrans players list HLS only, but the ASX
        # lookup returns a progressive MP4 (85-135x) -- try it first, HLS (~6x) after
        return "asx", hls
    return ("m3u8", hls) if hls else (None, None)


def transcript_of(vid):
    p = WHISPER / f"{vid}.json"
    return p if p.exists() else None


def opus_of(vid):
    p = OPUS / f"{vid}.opus"
    return p if p.exists() and p.stat().st_size > 0 else None


def fa_row(con, vid):
    return con.execute("SELECT video_id, status, last_error FROM videos WHERE video_id = ?", (ROW_PREFIX + vid,)).fetchone()


def hours_of(row):
    d = row.get("duration_s")
    if d:
        return float(d) / 3600
    kind, p = audio_on_disk(row["video_id"])
    if p:
        dd = ffprobe_duration(p)
        if dd:
            return dd / 3600
    return None


def scope(con, bodies=None, quiet=False):
    rows = meetings(con, bodies)
    out = []
    for r in rows:
        vid = r["video_id"]
        h = hours_of(r)
        kind, path = audio_on_disk(vid)
        ukind, url = audio_url(r)
        fa = fa_row(con, vid)
        excl = EXCLUDE.get(vid)
        if not excl and h is not None and h * 3600 < MIN_MEETING_S:
            excl = f"stub recording ({h * 60:.0f} min)"
        if not excl and fa and fa["status"] == "failed":
            excl = f"GPU transcript failed QA: {(fa['last_error'] or '')[:120]}"
        out.append({"video_id": vid, "body": r["speaker"], "date": r["meeting_date"], "hours": round(h, 2) if h is not None else None,
                    "status": r["status"], "captions": r["captions"], "platform": r["platform"],
                    "audio": kind, "audio_path": str(path) if path else None, "source": ukind, "opus": bool(opus_of(vid)),
                    "transcript": bool(transcript_of(vid)), "fa_status": fa["status"] if fa else None, "excluded": excl})
    jdump(SCOPE, {"at": state.now_iso(), "meetings": out})
    if not quiet:
        by = defaultdict(lambda: Counter())
        for m in out:
            b = by[m["body"]]
            b["n"] += 1; b["h"] += m["hours"] or 0
            b["unknown_h"] += int(m["hours"] is None)
            b["audio"] += int(bool(m["audio"])); b["opus"] += int(m["opus"]); b["transcript"] += int(m["transcript"])
            b["excluded"] += int(bool(m["excluded"]))
        print(f"{'body':11} {'mtgs':>4} {'hours':>7} {'?h':>3} {'audio':>5} {'opus':>4} {'trans':>5} {'excl':>4}")
        tot = Counter()
        for body in sorted(by):
            b = by[body]
            print(f"{body:11} {b['n']:4d} {b['h']:7.1f} {b['unknown_h']:3d} {b['audio']:5d} {b['opus']:4d} {b['transcript']:5d} {b['excluded']:4d}")
            tot.update(b)
        print(f"{'TOTAL':11} {tot['n']:4d} {tot['h']:7.1f} {tot['unknown_h']:3d} {tot['audio']:5d} {tot['opus']:4d} {tot['transcript']:5d} {tot['excluded']:4d}")
        for m in out:
            if m["excluded"]:
                print(f"  excluded {m['video_id']} ({m['body']}, {m['date']}, {m['hours']} h): {m['excluded']}")
    return out


# --- 2. audio ------------------------------------------------------------------------------

PROBE_S = 300.0            # seconds of audio measured for the polarity check
PROBE_SKIP_S = 300.0       # measured from here in, past an opening title card


def downmix_args(src, input_args=(), label=""):
    """The output args that turn `src` into ONE honest mono channel, plus the
    reason. Fremont's Granicus MP4s are polarity-inverted stereo (L = -R), so a
    plain `-ac 1` averages the channels and cancels the speech: it cost b17
    32 of 32 Fremont clips 32-38 dB (plan/ROADMAP.md) and it cost this route its
    first 24 Fremont meetings (b43, QA coverage 0-13 %). The detector is the
    clip path's own -- transcode.downmix_filter, channel 0 against the mono sum
    at POLARITY_DROP_DB -- so both routes decide this the same way.

    Returns (args, reason, ok). ok=False means the source could not be measured,
    and the caller must REFUSE rather than fall back to `-ac 1`: a silent
    fallback is exactly how 49 h of Fremont audio was cancelled twice."""
    base = list(input_args)
    p = transcode.probe(src, base)
    if not p:
        return None, "ffprobe cannot read the source", False
    if p["channels"] != 2:
        return ["-ac", "1"], f"{p['channels']}-channel source", True
    dur = p.get("duration") or 0.0
    meas = base + (["-ss", f"{PROBE_SKIP_S:.0f}"] if dur > PROBE_SKIP_S + PROBE_S else [])
    af, why = transcode.downmix_filter(src, PROBE_S, meas, probe_args=base)
    if af is None and why.startswith("channel probe failed"):
        return None, why, False
    if label:
        log(f"    {label}: {why}")
    return (["-af", af, "-ac", "1"] if af else ["-ac", "1"]), why, True


def ffmpeg_pull(url, dst, timeout):
    """One whole meeting -> mono 64 kbps opus, the cut master for step 5 (~29 MB
    per audio-hour; a 16 kHz FLAC of council audio measured 112 MB/h on
    gr-santaclara-2398, i.e. no compression, and 760 h of it would not fit this
    disk). The YouTube route cuts its clips from an opus master the same way
    (episodes.copy_cut), and register_clip re-encodes the clip to the 16 kHz /
    16 kbps corpus format. Returns (ok, detail, wall_s)."""
    part = Path(str(dst) + ".part.opus")
    t0 = time.time()
    ua = ["-user_agent", granicus.UA]
    mix, why, ok = downmix_args(url, ua, label="downmix")
    granicus._last_request["at"] = time.time()
    if not ok:
        return False, f"polarity check failed, refusing to downmix blind: {why}", time.time() - t0
    granicus.polite_wait()
    cmd = ["ffmpeg", "-nostdin", "-v", "error", "-y", *ua, "-i", url,
           "-vn", *mix, "-c:a", "libopus", "-b:a", "64k", str(part)]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        part.unlink(missing_ok=True)
        return False, f"ffmpeg timed out after {timeout:.0f} s", time.time() - t0
    if r.returncode != 0 or not part.exists() or part.stat().st_size == 0:
        part.unlink(missing_ok=True)
        return False, ((r.stderr or "").strip().splitlines() or [f"ffmpeg exit {r.returncode}"])[-1][:300], time.time() - t0
    part.replace(dst)
    return True, "ok", time.time() - t0


def to_opus(src, dst):
    tmp = Path(str(dst) + ".tmp.opus")
    mix, why, ok = downmix_args(src, label="gpu downmix")
    if not ok:
        return False, f"polarity check failed, refusing to downmix blind: {why}"
    r = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(src), *mix, "-ar", "16000",
                        "-c:a", "libopus", "-b:a", "16k", str(tmp)], capture_output=True, text=True, timeout=7200)
    if r.returncode != 0:
        tmp.unlink(missing_ok=True)
        return False, (r.stderr or "")[-200:]
    tmp.replace(dst)
    return True, f"{dst.stat().st_size / 1e6:.1f} MB"


def fetch(con, bodies=None, max_hours=None, limit=0):
    """Pull whole-meeting audio for every in-scope meeting without audio on disk.
    One ffmpeg at a time, >= GAP_S between requests (granicus.polite_wait), MP4
    before HLS, stop on 403/429 (the cutter's rule). Durations measured from the
    file land in scope.json (videos rows are not edited)."""
    AUDIO.mkdir(parents=True, exist_ok=True); OPUS.mkdir(parents=True, exist_ok=True)
    rows = [r for r in meetings(con, bodies) if r["video_id"] not in EXCLUDE]
    rows.sort(key=lambda r: (BODY_ORDER.index(r["speaker"]) if r["speaker"] in BODY_ORDER else 99, r["meeting_date"] or ""))
    done_h = 0.0
    n = 0
    fetched = []
    for r in rows:
        vid = r["video_id"]
        if audio_on_disk(vid)[0] or transcript_of(vid):
            continue
        h = hours_of(r)
        if h is not None and h * 3600 < MIN_MEETING_S:
            continue
        kind, url = audio_url(r)
        if kind == "asx":
            m = meta_of(r)
            hls = url
            granicus.polite_wait()
            try:
                url = granicus.fetch_archive_mp4(m.get("site") or vid.split("-")[1], m.get("view_id") or "2", m.get("clip_id") or vid.split("-")[-1])
            except Exception as e:                                   # noqa: BLE001
                log(f"  {vid}: ASX lookup failed: {type(e).__name__}: {str(e)[:120]}")
                url = None
            granicus._last_request["at"] = time.time()
            if url:
                kind = "mp4"
            elif hls:
                kind, url = "m3u8", hls
            else:
                log(f"  {vid} ({r['speaker']}): no media URL (ASX gave nothing, no HLS); skipped")
                continue
        if not url:
            log(f"  {vid} ({r['speaker']}): no m3u8 / mp4 in raw_meta; skipped")
            continue
        if max_hours and done_h + (h or 3.0) > max_hours:
            break
        if limit and n >= limit:
            break
        granicus.polite_wait()
        dst = AUDIO / f"{vid}.opus"
        timeout = max(3600.0, 3.0 * (h or 4.0) * 3600)
        log(f"  {vid} ({r['speaker']}, {r['meeting_date']}, {h if h is not None else '?'} h) via {kind} ...")
        ok, detail, wall = ffmpeg_pull(url, dst, timeout)
        granicus._last_request["at"] = time.time()
        n += 1
        if not ok:
            if HLS_BLOCKED.search(detail):
                log(f"STOP: the archive host refused {vid}: {detail}")
                fetched.append({"video_id": vid, "ok": False, "detail": detail, "blocked": True})
                break
            log(f"  {vid}: FAILED {detail} ({wall:.0f} s)")
            fetched.append({"video_id": vid, "ok": False, "detail": detail})
            continue
        dur = ffprobe_duration(dst) or 0.0
        done_h += dur / 3600
        ok2, d2 = to_opus(dst, OPUS / f"{vid}.opus")
        log(f"  {vid}: {dur / 3600:.2f} h in {wall:.0f} s ({dur / max(wall, 1):.0f}x), master {dst.stat().st_size / 1e6:.0f} MB, gpu opus {d2 if ok2 else 'FAILED ' + d2}")
        fetched.append({"video_id": vid, "ok": True, "hours": round(dur / 3600, 3), "wall_s": round(wall), "via": kind})
    flog = jload(LOGS / "fetch.json", [])
    flog += [dict(f, at=state.now_iso()) for f in fetched]
    LOGS.mkdir(parents=True, exist_ok=True)
    jdump(LOGS / "fetch.json", flog)
    log(f"fetch done: {sum(1 for f in fetched if f['ok'])} meetings, {done_h:.1f} h" + ("; STOPPED on 403/429" if any(f.get("blocked") for f in fetched) else ""))
    return 3 if any(f.get("blocked") for f in fetched) else 0


def encode(con, bodies=None):
    OPUS.mkdir(parents=True, exist_ok=True)
    n = 0
    for r in meetings(con, bodies):
        vid = r["video_id"]
        kind, p = audio_on_disk(vid)
        if not p or opus_of(vid) or transcript_of(vid) or vid in EXCLUDE:
            continue
        ok, d = to_opus(p, OPUS / f"{vid}.opus")
        log(f"  {vid}: {kind} -> opus {d if ok else 'FAILED ' + d}")
        n += ok
    log(f"encoded {n}")


def reuse(con):
    """The b35 whole-meeting transcripts (same configuration) for the four
    meetings that passed QA; nothing re-bought."""
    WHISPER.mkdir(parents=True, exist_ok=True)
    s = jload(RECALL / "sample.json", {"meetings": []})
    n = 0
    for m in s["meetings"]:
        vid = m["video_id"]
        src = Path(m.get("transcript") or "")
        if not src.exists() or m.get("ingest_status") != "done":
            log(f"  {vid}: not reusable ({m.get('ingest_status')})")
            continue
        dst = WHISPER / f"{vid}.json"
        if not dst.exists():
            shutil.copy2(src, dst)
            n += 1
        log(f"  {vid}: reused b35 transcript ({m.get('words'):,} words, {m.get('gpu_seconds')} GPU-s)")
    g = jload(GPU_LOG, {})
    g.setdefault("b35", {"source": "plan/RECALL-CHECK.md validation batch, reused", "meetings": [m["video_id"] for m in s["meetings"] if m.get("ingest_status") == "done"],
                         "hours": round(sum(m["audio_s"] for m in s["meetings"] if m.get("ingest_status") == "done") / 3600, 2),
                         "est_usd": s.get("gpu_est_usd"), "actual_usd": s.get("gpu_actual_usd"), "reused": True})
    jdump(GPU_LOG, g)
    log(f"reused {n} transcript(s)")


# --- 3. GPU: whole-meeting batches on the validation path ------------------------------------

def prepare(con, hours, bodies=None, dry_run=False):
    """Register 'fa-<id>' rows (kind 'validation', diarize False, the body's
    vocabulary) for meetings with an opus and no transcript, newest body-order
    first, up to `hours`, and ONE batch row awaiting approval. Refuses on the
    RunPod stop line exactly as gpu_batch.propose does."""
    c = gpu_batch.cfg()
    rows = [r for r in meetings(con, bodies) if r["video_id"] not in EXCLUDE]
    rows.sort(key=lambda r: (BODY_ORDER.index(r["speaker"]) if r["speaker"] in BODY_ORDER else 99, r["meeting_date"] or ""))
    pick, total = [], 0.0
    for r in rows:
        vid = r["video_id"]
        op = opus_of(vid)
        if not op or transcript_of(vid):
            continue
        fa = fa_row(con, vid)
        if fa and fa["status"] in ("queued", "transcribing", "done", "failed"):
            continue                                      # in a batch, or already judged
        dur = ffprobe_duration(op) or float(r.get("duration_s") or 0)
        if dur < MIN_MEETING_S:
            continue
        if total + dur / 3600 > hours:
            continue
        pick.append((r, op, dur))
        total += dur / 3600
    if not pick:
        log("nothing to prepare (no opus without a transcript)")
        return None
    est = gpu_batch.estimate_usd(total, c)
    why = spend.runpod_refusal(con, est)
    log(f"prepare: {len(pick)} meetings, {total:.2f} audio-h, GPU est ${est:.2f} (RTFX {c['rtfx']:.0f}x ESTIMATE, ${c['usd_hr']:.2f}/h, "
        f"x{c['contingency']:.1f}; measured no-diarization rate $0.0102/h -> ~${total * 0.0102:.2f}); " + ", ".join(f"{r['speaker']}:{sum(1 for x in pick if x[0]['speaker'] == r['speaker'])}" for r in {x[0]['speaker']: x[0] for x in pick}.values()))
    if why:
        log(f"REFUSED by the RunPod stop line: {why}")
        return None
    if dry_run:
        for r, op, dur in pick:
            log(f"  {r['video_id']:22} {r['speaker']:11} {r['meeting_date']} {dur / 3600:5.2f} h")
        return None
    now = state.now_iso()
    for r, op, dur in pick:
        vid, rid = r["video_id"], ROW_PREFIX + r["video_id"]
        meta = {"validation": "fullasr", "source_video_id": vid, "meeting_id": r["meeting_id"], "body": r["speaker"],
                "initial_prompt": vocab.for_body(con, r["speaker"]), "diarize": False, "note": GPU_NOTE}
        if fa_row(con, vid):
            con.execute("UPDATE videos SET opus_path = ?, transcribed_s = ?, duration_s = ?, raw_meta = ?, updated_at = ?, status = 'transcoded' WHERE video_id = ?",
                        (str(op), round(dur, 1), round(dur, 1), json.dumps(meta), now, rid))
        else:
            con.execute("INSERT INTO videos (video_id, speaker, channel_url, month, title, duration_s, raw_meta, crawled_at, in_frame, "
                        "status, opus_path, transcribed_s, attempts, updated_at, kind, platform, source_url, meeting_id, meeting_date) "
                        "VALUES (?,?,?,?,?,?,?,?,0,'transcoded',?,?,0,?,'validation',?,?,?,?)",
                        (rid, r["speaker"], r["channel_url"] or r["source_url"] or "", (r["meeting_date"] or now)[:7],
                         f"[fullasr] {r['title'] or vid}", round(dur, 1), json.dumps(meta), now, str(op), round(dur, 1), now,
                         r["platform"], r["source_url"], r["meeting_id"], r["meeting_date"]))
    bid = gpu_batch.next_batch_id(con)
    con.execute("BEGIN IMMEDIATE")
    con.execute("INSERT INTO batches (batch_id, created_at, status, n_videos, audio_hours, est_usd, approval_requested_at, note) VALUES (?,?,?,?,?,?,?,?)",
                (bid, now, "awaiting_approval", len(pick), round(total, 3), est, now,
                 f"{GPU_NOTE}; est = ({total:.2f} h / {c['rtfx']:.0f}x + {c['boot_min']:.0f} min) * ${c['usd_hr']:.2f}/h * {c['contingency']:.1f}"))
    con.executemany("INSERT INTO batch_videos (batch_id, video_id) VALUES (?, ?)", [(bid, ROW_PREFIX + r["video_id"]) for r, _, _ in pick])
    con.execute("COMMIT")
    for r, _, _ in pick:
        state.transition(con, ROW_PREFIX + r["video_id"], "transcoded", "queued", note=f"batch {bid} ({GPU_NOTE})")
    g = jload(GPU_LOG, {})
    g[bid] = {"created_at": now, "meetings": [r["video_id"] for r, _, _ in pick], "hours": round(total, 3), "est_usd": est,
              "est_measured_rate_usd": round(total * 0.0102, 3), "bodies": dict(Counter(r["speaker"] for r, _, _ in pick))}
    jdump(GPU_LOG, g)
    log(f"batch {bid}: {len(pick)} whole meetings, {total:.2f} audio-h, est ${est:.2f}; awaiting_approval.")
    log(f"next: .venv/bin/python pipeline/gpu_batch.py approve {bid} --via fullasr ; then nohup launch with "
        f"WHISPER_DIR={WHISPER} BATCH_DIR={BATCHES} NOTIFY_BACKEND=none")
    return bid


def harvest(con, bid):
    """fa-<id>.json (ingested 'done' by gpu_job) -> whisper/<id>.json; QA-failed
    rows recorded (their files stay in batches/<bid>/out/, never loaded)."""
    WHISPER.mkdir(parents=True, exist_ok=True)
    b = dict(con.execute("SELECT * FROM batches WHERE batch_id = ?", (bid,)).fetchone())
    got, failed = [], {}
    for r in con.execute("SELECT v.video_id, v.status, v.last_error, v.transcribed_s FROM batch_videos b JOIN videos v USING (video_id) WHERE b.batch_id = ?", (bid,)):
        rid = r["video_id"]
        vid = rid[len(ROW_PREFIX):] if rid.startswith(ROW_PREFIX) else rid
        src = WHISPER / f"{rid}.json"
        if r["status"] == "done" and src.exists():
            dst = WHISPER / f"{vid}.json"
            if not dst.exists():
                shutil.copy2(src, dst)
            doc = json.loads(dst.read_text())
            words = sum(len(x["text"].split()) for x in doc["segments"])
            got.append(vid)
            log(f"  {vid}: {len(doc['segments'])} segments, {words:,} words, {words / max(float(r['transcribed_s'] or 1) / 3600, 0.01):.0f} words/h, {doc.get('gpu_seconds')} GPU-s")
        else:
            failed[vid] = f"{r['status']}: {(r['last_error'] or '')[:200]}"
            log(f"  {vid}: NOT harvested -- {failed[vid]}")
    g = jload(GPU_LOG, {})
    g.setdefault(bid, {})
    g[bid].update({"status": b["status"], "actual_usd": b.get("actual_usd"), "gpu_minutes": b.get("gpu_minutes"), "harvested": got,
                   "failed": failed, "note": (b.get("note") or "")[:300], "est_usd": b.get("est_usd"), "hours": b.get("audio_hours")})
    jdump(GPU_LOG, g)
    log(f"harvested {len(got)}, failed {len(failed)}; batch {bid} {b['status']}, {b.get('gpu_minutes')} GPU-min, actual ${b.get('actual_usd')} vs est ${b.get('est_usd')}")


FILLER_MAX_WORDS = 4        # "you", "Thank you.", ".", "We'll be right back." -- Whisper over dead air / a recess bumper
LOOP_RULE = "(decoder loop)"


def filler_runs(segs):
    """(i, j) index runs of >= 2 identical consecutive segments whose text is
    <= FILLER_MAX_WORDS words (Whisper over dead air), in segment order."""
    runs, i = [], 0
    while i < len(segs):
        j, t = i, (segs[i].get("text") or "").strip().lower()
        while j + 1 < len(segs) and (segs[j + 1].get("text") or "").strip().lower() == t:
            j += 1
        if j > i and t and len(t.split()) <= FILLER_MAX_WORDS:
            runs.append((i, j))
        i = j + 1
    return runs


def rescue(con, bid, dry_run=False):
    """b40 (2026-10-18): 14 of 24 Richmond whole meetings FAILED the gate's
    'identical segments spanning > DUP_FAIL_S' rule (analysis/qa_transcripts.py)
    on runs of 'you' / 'Thank you.' / '.' over recesses and closed sessions
    (2-7 segments spanning 2-89 min of dead air; coverage 39-70 %, 3,900-7,100
    words/h otherwise). That is Whisper's dead-air hallucination, not a decoder
    loop over speech: the rule's span unit was chosen for CLIPS, where a 60-s
    run is speech overwritten; a 5-7 h council meeting has a recess in it as a
    rule. A failed fa- row is loaded here ONLY IF (a) its only FAIL is that
    rule, (b) every run the rule could have fired on is <= FILLER_MAX_WORDS
    words, and (c) the transcript with each such run collapsed to its first
    segment passes the whole gate again (coverage >= 30 %, cps, compression).
    The dropped segments carried no words (the locate pass skips windows under
    MIN_WORDS anyway). Nothing is written to output/whisper/ or the corpus:
    this route's transcript is a locate input; the diarized clips cut from it
    pass the gate on their own. Recorded in gpu_batches.json[bid]['rescued'],
    the fa- row 'failed' -> 'done' with the reason in its events row, the
    rescued file carries a top-level 'rescued' key, and plan/FULLASR-GRANICUS.md
    §4 lists every one. Real failures (coverage < 30 %, runs of sentences)
    stay failed."""
    sys.path.insert(0, str(ROOT / "analysis"))
    import qa_transcripts as qa                   # noqa: E402
    staging = BATCHES / bid / "out"
    g = jload(GPU_LOG, {})
    g.setdefault(bid, {})
    rescued, kept = g[bid].setdefault("rescued", {}), {}
    for r in con.execute("SELECT v.video_id, v.status, v.last_error FROM batch_videos b JOIN videos v USING (video_id) WHERE b.batch_id = ? AND v.status = 'failed'", (bid,)):
        rid, err = r["video_id"], r["last_error"] or ""
        vid = rid[len(ROW_PREFIX):] if rid.startswith(ROW_PREFIX) else rid
        src = staging / f"{rid}.json"
        fails = [x.strip() for x in err.replace("QA FAIL: ", "", 1).split(";") if x.strip()]
        if not src.exists() or not fails or any(LOOP_RULE not in f for f in fails):
            kept[vid] = f"not the loop rule alone: {err[:120]}"
            log(f"  {vid}: stays failed -- {kept[vid]}")
            continue
        doc = json.loads(src.read_text(encoding="utf-8"))
        segs = doc["segments"]
        runs = filler_runs(segs)
        drop = {k for i, j in runs for k in range(i + 1, j + 1)}
        if not runs:
            kept[vid] = "loop rule fired on a run of real text, not filler"
            log(f"  {vid}: stays failed -- {kept[vid]}")
            continue
        new = dict(doc)
        new["segments"] = [s for k, s in enumerate(segs) if k not in drop]
        new["rescued"] = {"at": state.now_iso(), "rule": f"dead-air filler runs collapsed (<= {FILLER_MAX_WORDS} words, scripts/fullasr.py rescue)",
                          "original_fail": err[:300], "segments_removed": len(drop),
                          "runs": [{"start_s": round(segs[i]["start"], 1), "end_s": round(segs[j]["end"], 1), "n": j - i + 1, "text": (segs[i].get("text") or "").strip()[:40]} for i, j in runs]}
        tmp = staging / f"{rid}.rescue.json"
        tmp.write_text(json.dumps(new, ensure_ascii=False), encoding="utf-8")
        q = qa.check_video(tmp)
        if q["fail"]:
            kept[vid] = "still FAILS with filler collapsed: " + "; ".join(q["fail"])[:200]
            tmp.unlink(missing_ok=True)
            log(f"  {vid}: stays failed -- {kept[vid]}")
            continue
        words = sum(len(s["text"].split()) for s in new["segments"])
        span = max(s["end"] for s in new["segments"])
        widest = max((segs[j]["end"] - segs[i]["start"]) for i, j in runs)
        summary = (f"{len(runs)} filler run(s), {len(drop)} segments dropped, widest {widest / 60:.0f} min of dead air; "
                   f"{words:,} words, {words / (span / 3600):.0f} words/h, coverage {q.get('coverage', 0):.0%}" + (f"; QA WARN: {'; '.join(q['warn'])[:200]}" if q["warn"] else ""))
        if dry_run:
            tmp.unlink(missing_ok=True)
            rescued[vid] = {"dry_run": True}
            log(f"  {vid}: WOULD rescue -- {summary}")
            continue
        WHISPER.mkdir(parents=True, exist_ok=True)
        shutil.move(str(tmp), WHISPER / f"{rid}.json")
        shutil.copy2(WHISPER / f"{rid}.json", WHISPER / f"{vid}.json")
        ok = state.transition(con, rid, "failed", "done", note=f"rescued by scripts/fullasr.py rescue: {summary}"[:300],
                              transcribed_s=doc.get("transcribed_s"), last_error=None)
        rescued[vid] = {"at": state.now_iso(), "original_fail": err[:200], "summary": summary, "row_updated": bool(ok)}
        if vid in g[bid].get("failed", {}):
            g[bid]["failed"].pop(vid)
        g[bid].setdefault("harvested", [])
        if vid not in g[bid]["harvested"]:
            g[bid]["harvested"].append(vid)
        log(f"  {vid}: RESCUED -- {summary}")
    if not dry_run:
        g[bid]["still_failed"] = kept
        jdump(GPU_LOG, g)
    log(f"rescue {bid}: {len(rescued)} rescued, {len(kept)} stay failed{' (dry run)' if dry_run else ''}")


# --- 4. Sonnet v2.1, no screen -------------------------------------------------------------

def segments_of(doc):
    return [{"start": float(x["start"]), "end": float(x["end"]), "text": x["text"]} for x in doc["segments"]]


def build_windows(vids=None):
    """Every 120-s window of each transcript, padded +- 30 s like
    classify.candidates (recall_fullasr_v2.build_windows)."""
    WINDOWS.mkdir(parents=True, exist_ok=True)
    n_new = 0
    for p in sorted(WHISPER.glob("*.json")):
        vid = p.stem
        if vid.startswith(ROW_PREFIX) or (vids and vid not in vids):
            continue
        dst = WINDOWS / f"{vid}.jsonl"
        if dst.exists():
            continue
        segs = segments_of(json.loads(p.read_text()))
        end = max((s["end"] for s in segs), default=0)
        rows = []
        for i in range(int(end // WINDOW) + 1):
            a, b = i * WINDOW, (i + 1) * WINDOW
            pa, pb = max(0.0, a - PAD), min(end, b + PAD)
            text = cl.window_text(segs, pa, pb)
            rows.append({"id": f"{CALL_PREFIX}{vid}_w{i:04d}", "video_id": vid, "i": i, "start_s": a, "end_s": b, "pad_start_s": pa, "pad_end_s": pb,
                         "words": len(text.split()), "text": text, "label": None})
        dst.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
        n_new += 1
        log(f"  {vid}: {len(rows)} windows, {sum(1 for r in rows if r['words'] >= MIN_WORDS)} with >= {MIN_WORDS} words ({end / 3600:.2f} h)")
    log(f"windows: {n_new} new meeting(s)")


def read_windows(vid):
    p = WINDOWS / f"{vid}.jsonl"
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


def labelled(con, vid):
    """{call id: row} of labelled v2.1 calls for this meeting (this route)."""
    return {r["id"]: dict(r) for r in con.execute("SELECT * FROM classifier_calls WHERE video_id = ? AND prompt_version = ? AND label IS NOT NULL",
                                                   (CALL_PREFIX + vid, pv()))}


def pending(con, vid):
    return {r["id"] for r in con.execute("SELECT id FROM classifier_calls WHERE video_id = ? AND prompt_version = ? AND label IS NULL AND batch_id LIKE 'msgbatch%'",
                                         (CALL_PREFIX + vid, pv()))}


def submit(con, cap_usd, bodies=None, max_windows=0, dry_run=False):
    build_windows()
    book_ver = cl.codebook_version_for(CODEBOOK)
    prompt = pv()
    scope_bodies = {m["video_id"]: m["speaker"] for m in meetings(con, bodies)}
    send = []
    per = {}
    for p in sorted(WINDOWS.glob("*.jsonl")):
        vid = p.stem
        if vid not in scope_bodies:
            continue
        have, pend = labelled(con, vid), pending(con, vid)
        todo = [r for r in read_windows(vid) if r["words"] >= MIN_WORDS and r["id"] not in have and r["id"] not in pend]
        per[vid] = (len(todo), len(have), len(pend))
        send += todo
    if max_windows:
        send = send[:max_windows]
    tokens = sum(cl.caption_tokens_for(r["words"]) for r in send)
    est = cl.estimate_usd(tokens, len(send), MODEL, batch=True, codebook=CODEBOOK)
    worst = sum(cl.worst_case_usd(r["text"], MODEL, batch=True, codebook=CODEBOOK) for r in send)
    log(f"submit: {len(send)} windows over {sum(1 for v in per.values() if v[0])} meeting(s) ({sum(v[1] for v in per.values())} labelled, "
        f"{sum(v[2] for v in per.values())} pending); codebook {book_ver}, prompt_version {prompt}; estimate ${est:.2f} (worst ${worst:.2f}) "
        f"at the batch rate; cap ${cap_usd}; Claude in DB ${spend.claude_total(con):.2f} of ${spend.caps()['CLAUDE_CAP_USD']:.0f}")
    if not send:
        return 0
    if worst > cap_usd:
        log(f"REFUSED: worst case ${worst:.2f} exceeds --cap-usd ${cap_usd}; use --max-windows")
        return 2
    why = spend.claude_refusal(con, worst)
    if why:
        log(f"REFUSED: {why}")
        return 2
    if dry_run:
        log("dry run: nothing submitted")
        return 0
    import anthropic
    client = anthropic.Anthropic()
    now = state.now_iso()
    provisional = f"pending-fullasr-{now}"
    con.execute("INSERT INTO classifier_batches (batch_id, created_at, status, model, prompt_version, n_requests, est_usd, note) VALUES (?,?,?,?,?,?,?,?)",
                (provisional, now, BATCH_STATUS, MODEL, prompt, len(send), round(est, 4), BATCH_NOTE))
    reqs = []
    for r in send:
        c = {"id": r["id"], "start_s": r["start_s"], "end_s": r["end_s"], "pad_start_s": r["pad_start_s"], "pad_end_s": r["pad_end_s"],
             "words": r["words"], "hits_per_100w": None, "family_rates": {}, "top_keywords": [], "keywords_version": "none"}
        rec = cl.call_record(c, CALL_PREFIX + r["video_id"], MODEL, prompt, batch_id=provisional)
        cl.insert_call(con, cl.call_row(rec), replace=True)
        reqs.append({"custom_id": API_ID_PREFIX + r["id"][len(CALL_PREFIX):], "params": cl.request_params(MODEL, r["text"], cl.DEFAULT_LANG, codebook=CODEBOOK)})
    try:
        mb = client.messages.batches.create(requests=reqs)
    except Exception as e:                                        # noqa: BLE001 -- nothing billed; undo
        con.execute("DELETE FROM classifier_calls WHERE batch_id = ? AND label IS NULL", (provisional,))
        con.execute("DELETE FROM classifier_batches WHERE batch_id = ?", (provisional,))
        log(f"batch create failed: {type(e).__name__}: {str(e)[:200]}; provisional rows removed")
        return 1
    con.execute("UPDATE classifier_batches SET batch_id = ?, note = ? WHERE batch_id = ?",
                (mb.id, f"{BATCH_NOTE}; codebook {book_ver}; {len(send)} windows; processing_status {mb.processing_status}", provisional))
    con.execute("UPDATE classifier_calls SET batch_id = ? WHERE batch_id = ?", (mb.id, provisional))
    LOGS.mkdir(parents=True, exist_ok=True)
    sl = jload(LOGS / "sonnet_batches.json", [])
    sl.append({"batch_id": mb.id, "submitted_at": now, "n_requests": len(reqs), "codebook_version": book_ver, "prompt_version": prompt,
               "est_usd": round(est, 4), "worst_usd": round(worst, 4), "meetings": sorted({r["video_id"] for r in send})})
    jdump(LOGS / "sonnet_batches.json", sl)
    log(f"submitted {mb.id}: {len(reqs)} requests, status {mb.processing_status}")
    return 0


def collect(con, no_derive=False):
    import anthropic
    client = anthropic.Anthropic()
    batches = [dict(r) for r in con.execute("SELECT * FROM classifier_batches WHERE status = ? ORDER BY created_at", (BATCH_STATUS,))]
    if not batches:
        log("no fullasr batches pending")
    touched = set()
    for b in batches:
        bid, model, prompt = b["batch_id"], b["model"], b["prompt_version"]
        if not bid.startswith("msgbatch"):
            continue
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
                if cl.apply_result(con, bid, model, prompt, cid, res, usage, cost):
                    ok += 1
                    flagged += int(bool(res.get("contains_instructions")))
                else:
                    bad += 1
            else:
                bad += 1
                err = getattr(getattr(result.result, "error", None), "type", result.result.type)
                cl.apply_result(con, bid, model, prompt, cid, None, None, 0.0, error=f"batch {result.result.type}: {err}")
            touched.add(cl.video_of_call_id(cid)[len(CALL_PREFIX):])
        con.execute("UPDATE classifier_batches SET status = 'collected', n_succeeded = ?, n_errored = ?, cost_usd = ?, collected_at = ? WHERE batch_id = ?",
                    (ok, bad, round(cost_total, 4), state.now_iso(), bid))
        sl = jload(LOGS / "sonnet_batches.json", [])
        for s in sl:
            if s["batch_id"] == bid:
                s.update({"collected_at": state.now_iso(), "n_succeeded": ok, "n_errored": bad, "actual_usd": round(cost_total, 4), "flagged": flagged})
        jdump(LOGS / "sonnet_batches.json", sl)
        log(f"{bid}: {ok} labels, {bad} errored, ${cost_total:.4f}; {flagged} window(s) flagged instruction-like text" + (" -- TELL PUCK" if flagged else ""))
    if touched and not no_derive:
        derive(con, sorted(touched))
    return 0


# --- 5. episodes (method fullasr) and offline cuts ----------------------------------------------

def sibling_mid_cut(con, row):
    """A meeting the sibling worker is fetching or cutting right now: never
    re-finish it (AGENTS / task rule). 'downloaded' with located episodes of
    another route = its cutter has claimed it and has work left."""
    # The task's rule is the status pair below. A 'downloaded' row with located
    # rows of another route is the sibling's re-read waiting for ITS cutter, not
    # a cut in progress; this route writes only '_f' rows and cuts them in the
    # same call, so the two never touch the same episode or the meeting status.
    return row["status"] in ("downloading", "cutting")


def index_points_of(row):
    vid = row["video_id"]
    for cand in ([Path(row["captions_path"]).with_name(f"{vid}.player.json")] if row.get("captions_path") else []) + [SUBS / f"{vid}.player.json"]:
        if cand.exists():
            try:
                return [(float(t), title) for t, title in json.loads(cand.read_text(encoding="utf-8")).get("index_points") or []]
            except (ValueError, TypeError):
                return None
    return None


def write_fullasr_episodes(con, row, calls, index_points=None):
    """episodes rows for one meeting from its fullasr v2.1 calls, modelled on
    classify.write_episodes with this route's rules made explicit:
      * located runs -> '<vid>_f<n>', mention runs -> '<vid>_fm<n>' (status 'mention');
      * a span whose core already has a row OF THIS ROUTE is that row (updated while
        located/mention); rows of other routes are never updated (v1 untouched);
      * a new located span is trimmed to its largest piece >= 60 s outside every
        span of ANY route that is cut / queued / transcribing / done (the clip
        exists or will), or dropped -- already-transcribed v1 clips are kept and
        pointed at in the events note;
      * located/mention rows of THIS route no span matches are superseded
        (events row), never any other route's row.
    Returns (located, mentions, n_written, trimmed, kept_refs)."""
    vid, meeting_id = row["video_id"], row["meeting_id"]
    rows = []
    for r in calls.values():
        rr = dict(r); rr["video_id"] = vid
        rows.append(rr)
    located = cl.merge_episodes(rows, GAP, cl.ABOUT, "f")
    mentions = cl.merge_episodes(rows, GAP, {"mention_only"}, "fm")
    by_id = {r["id"]: r for r in rows}
    now = state.now_iso()
    existing = {r["episode_id"]: dict(r) for r in con.execute(
        "SELECT episode_id, status, method, core_start_s, core_end_s, start_s, end_s, transcript_path, opus_path, audio_path "
        "FROM episodes WHERE video_id = ?", (vid,))}
    core = lambda r: ((r["core_start_s"] if r["core_start_s"] is not None else r["start_s"]),   # noqa: E731
                      (r["core_end_s"] if r["core_end_s"] is not None else r["end_s"]))
    by_span = {}
    for eid, r in existing.items():
        if r["method"] == METHOD:
            a0, b0 = core(r)
            by_span.setdefault((int(round(a0)), int(round(b0))), eid)
    max_n = {"f": 0, "fm": 0}
    for eid in existing:
        m = re.search(r"_(fm|f)(\d+)$", eid)
        if m:
            max_n[m.group(1)] = max(max_n[m.group(1)], int(m.group(2)))
    fixed = [(core(r), eid) for eid, r in existing.items() if r["status"] not in ("located", "mention", "failed")]
    fixed_spans = [s for s, _ in fixed]
    trimmed, kept_refs = [], []
    n = 0
    con.execute("BEGIN IMMEDIATE")
    try:
        for e, status, prefix in [(x, "located", "f") for x in located] + [(x, "mention", "fm") for x in mentions]:
            full = (e["start_s"], e["end_s"])
            same = by_span.get((int(round(e["start_s"])), int(round(e["end_s"]))))
            if same:
                e["episode_id"] = same
            else:
                if status == "located" and fixed_spans:
                    piece = cl.largest_uncovered(e["start_s"], e["end_s"], fixed_spans)
                    covering = [eid for (s, t), eid in fixed if s < full[1] and t > full[0]]
                    if piece is None:
                        trimmed.append((e["episode_id"], full, None, covering))
                        kept_refs.append({"span": full, "covered_by": covering, "piece": None})
                        # PROPOSED-14 4: the v2.1 row points at the existing transcript. Written
                        # as a 'done' fullasr row over its own core span only when every covering
                        # clip is 'done' (transcript_path = the first covering clip's); otherwise
                        # the events row records the span and nothing is cut twice.
                        cov_rows = [existing[c] for c in covering if c in existing]
                        if cov_rows and all(r["status"] == "done" and r.get("transcript_path") for r in cov_rows):
                            if e["episode_id"] in existing:
                                max_n[prefix] += 1
                                e["episode_id"] = f"{vid}_{prefix}{max_n[prefix]}"
                            else:
                                m = re.search(r"_(fm|f)(\d+)$", e["episode_id"])
                                if m:
                                    max_n[prefix] = max(max_n[prefix], int(m.group(2)))
                            agenda = granicus.agenda_item_for(index_points, e["start_s"], e["end_s"]) if index_points else None
                            cur = con.execute(
                                "INSERT OR IGNORE INTO episodes (episode_id, meeting_id, video_id, start_s, end_s, core_start_s, core_end_s, tags, method, "
                                "score, label, classifier_model, prompt_version, agenda_item, status, attempts, updated_at, keywords_version, "
                                "transcript_path, opus_path, audio_path) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,'done',0,?,?,?,?,?)",
                                (e["episode_id"], meeting_id, vid, e["start_s"], e["end_s"], e["start_s"], e["end_s"], cl.TAGS.get(e["label"], "both"), METHOD,
                                 None, e["label"], MODEL, pv(), agenda, now, "none", cov_rows[0]["transcript_path"], cov_rows[0].get("opus_path"), cov_rows[0].get("audio_path")))
                            n += cur.rowcount
                            existing[e["episode_id"]] = {"status": "done", "method": METHOD, "core_start_s": e["start_s"], "core_end_s": e["end_s"],
                                                         "start_s": e["start_s"], "end_s": e["end_s"]}
                            by_span.setdefault((int(round(e["start_s"])), int(round(e["end_s"]))), e["episode_id"])
                            con.execute("INSERT INTO events (video_id, from_status, to_status, at, note) VALUES (?,?,?,?,?)",
                                        (e["episode_id"], None, "done", now, f"fullasr span {hms(full[0])}-{hms(full[1])} fully covered by transcribed "
                                         f"{' '.join(covering)[:120]}; points at that transcript, nothing cut"[:300]))
                        continue
                    if piece != full:
                        trimmed.append((e["episode_id"], full, piece, covering))
                        kept_refs.append({"span": full, "covered_by": covering, "piece": piece})
                        e["start_s"], e["end_s"] = piece
                        e["duration_s"] = round(piece[1] - piece[0], 1)
                        same = by_span.get((int(round(piece[0])), int(round(piece[1]))))
                        if same:
                            e["episode_id"] = same
                if not same and e["episode_id"] in existing:
                    max_n[prefix] += 1
                    e["episode_id"] = f"{vid}_{prefix}{max_n[prefix]}"
                else:
                    m = re.search(r"_(fm|f)(\d+)$", e["episode_id"])
                    if m:
                        max_n[prefix] = max(max_n[prefix], int(m.group(2)))
            existing.setdefault(e["episode_id"], {"status": status, "method": METHOD, "core_start_s": e["start_s"], "core_end_s": e["end_s"],
                                                  "start_s": e["start_s"], "end_s": e["end_s"]})
            by_span.setdefault((int(round(e["start_s"])), int(round(e["end_s"]))), e["episode_id"])
            tags = cl.TAGS.get(e["label"], "both")
            if status == "mention":
                tags = "both"
            agenda = granicus.agenda_item_for(index_points, e["start_s"], e["end_s"]) if index_points else None
            cur = con.execute(
                "INSERT INTO episodes (episode_id, meeting_id, video_id, start_s, end_s, core_start_s, core_end_s, tags, method, "
                "score, label, classifier_model, prompt_version, agenda_item, status, attempts, updated_at, keywords_version) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,0,?,?) "
                "ON CONFLICT(episode_id) DO UPDATE SET start_s = excluded.start_s, end_s = excluded.end_s, "
                "core_start_s = excluded.core_start_s, core_end_s = excluded.core_end_s, tags = excluded.tags, "
                "score = excluded.score, label = excluded.label, classifier_model = excluded.classifier_model, "
                "prompt_version = excluded.prompt_version, agenda_item = excluded.agenda_item, updated_at = excluded.updated_at, "
                "keywords_version = excluded.keywords_version "
                "WHERE episodes.status IN ('located', 'mention') AND episodes.method = excluded.method",
                (e["episode_id"], meeting_id, vid, e["start_s"], e["end_s"], e["start_s"], e["end_s"], tags, METHOD,
                 None, e["label"], MODEL, pv(), agenda, status, now, "none"))
            n += cur.rowcount
            if full != (e["start_s"], e["end_s"]):
                con.execute("INSERT INTO events (video_id, from_status, to_status, at, note) VALUES (?,?,?,?,?)",
                            (e["episode_id"], None, status, now, f"fullasr span {hms(full[0])}-{hms(full[1])} overlaps transcribed clip(s) "
                             f"{' '.join(c for _, s, _, c in [(0, 0, 0, [x for (s, t), x in fixed if s < full[1] and t > full[0]])] for c in c)[:150]}; "
                             f"kept those, cut only {hms(e['start_s'])}-{hms(e['end_s'])}"[:300]))
        kept = {e["episode_id"] for e in located} | {e["episode_id"] for e in mentions}
        stale = [eid for eid, r in existing.items() if r.get("method") == METHOD and r["status"] in ("located", "mention")
                 and eid not in kept and re.search(r"_(fm|f)\d+$", eid)
                 and con.execute("SELECT 1 FROM episodes WHERE episode_id = ?", (eid,)).fetchone()]
        for eid in stale:
            con.execute("DELETE FROM episodes WHERE episode_id = ? AND status IN ('located', 'mention') AND method = ?", (eid, METHOD))
            con.execute("INSERT INTO events (video_id, from_status, to_status, at, note) VALUES (?,?,?,?,?)",
                        (eid, existing[eid]["status"], "superseded", now, "fullasr re-read: no span of the current reading matches this row; removed"))
        con.execute("COMMIT")
    except Exception:
        con.execute("ROLLBACK")
        raise
    for eid, full, piece, covering in trimmed:
        log(f"  {vid}: fullasr span {hms(full[0])}-{hms(full[1])} overlaps transcribed {' '.join(covering)[:80]}; "
            + (f"cut only {hms(piece[0])}-{hms(piece[1])}" if piece else "nothing uncovered >= 60 s, kept the v1 transcript"))
    return located, mentions, n, trimmed, kept_refs


def cut_meeting(con, row, log=log):
    """Cut every 'located' fullasr episode of one meeting from the whole-meeting
    audio on disk (offline, no request; the local_wav path generalised): core
    +- 30 s, clamped, snapped to the WhisperX segment edges, -c copy, then
    register_clip -> output/opus/<eid>.opus + an in-frame 'transcoded' videos row.
    Returns (n_ok, n_fail, seconds)."""
    vid = row["video_id"]
    kind, src = audio_on_disk(vid)
    if not src:
        log(f"  {vid}: no whole-meeting audio on disk to cut from")
        return 0, 0, 0.0
    doc = jload(WHISPER / f"{vid}.json")
    edges = epm.cue_edges(segments_of(doc)) if doc else []
    dur_full = ffprobe_duration(src) or row["duration_s"]
    CLIPS.mkdir(parents=True, exist_ok=True)
    opus_dir = transcode.opus_dir()
    opus_dir.mkdir(parents=True, exist_ok=True)
    ok = bad = 0
    secs = 0.0
    eps = [dict(r) for r in con.execute("SELECT * FROM episodes WHERE video_id = ? AND status = 'located' AND method = ? ORDER BY start_s", (vid, METHOD))]
    for ep in eps:
        if con.execute("SELECT 1 FROM videos WHERE video_id = ?", (ep["episode_id"],)).fetchone():
            msg, _ = epm.register_clip(con, row, ep, None, ep["start_s"], ep["end_s"], opus_dir)
            epm.episode_status_from_clip(con, ep["episode_id"])
            log(f"  {ep['episode_id']}: {msg}")
            ok += 1
            continue
        a, b = epm.clip_bounds(ep["core_start_s"], ep["core_end_s"], dur_full, edges)
        clip = CLIPS / f"{ep['episode_id']}.{kind}"
        if not clip.exists():
            good, how, d = epm.copy_cut(src, a, b, clip)
            if not good:
                state.set_episode(con, ep["episode_id"], "failed", note=f"cut (fullasr audio): {how}", last_error=str(how)[:300])
                log(f"  {ep['episode_id']}: cut FAILED {how}")
                bad += 1
                continue
        else:
            how, d = "on disk", ffprobe_duration(clip)
        msg, clip_id = epm.register_clip(con, row, ep, clip, a, b, opus_dir, cut_source=CUT_SOURCE)
        if not clip_id:
            state.set_episode(con, ep["episode_id"], "failed", note=msg[:200], last_error=msg[:300])
            log(f"  {ep['episode_id']}: {msg}")
            bad += 1
            continue
        state.set_episode(con, ep["episode_id"], "cut", note=f"fullasr audio ({kind}), {how}, {d:.1f} s", start_s=a, end_s=b,
                          audio_path=str(clip), opus_path=str(opus_dir / f"{clip_id}.opus"))
        secs += d or 0
        ok += 1
        log(f"  {ep['episode_id']} {hms(a)}-{hms(b)} ({(b - a) / 60:.1f} min, {kind} {how}): {msg}")
    return ok, bad, secs


def derive(con, vids=None, no_cut=False):
    """Meetings whose windows are all labelled -> episodes (method fullasr) -> cuts."""
    rows = {m["video_id"]: m for m in meetings(con)}
    todo = sorted(vids) if vids else sorted(p.stem for p in WINDOWS.glob("*.jsonl"))
    summary = jload(OUT / "derived.json", {})
    for vid in todo:
        row = rows.get(vid)
        if not row:
            continue
        wins = [w for w in read_windows(vid) if w["words"] >= MIN_WORDS]
        have = labelled(con, vid)
        if not wins or any(w["id"] not in have for w in wins):
            missing = sum(1 for w in wins if w["id"] not in have)
            if wins:
                log(f"  {vid}: {missing} window(s) still unlabelled; not derived")
            continue
        if sibling_mid_cut(con, row):
            log(f"  {vid} ({row['speaker']}): sibling mid-cut (status {row['status']}); episodes deferred")
            summary[vid] = {"deferred": f"sibling mid-cut, status {row['status']}", "at": state.now_iso()}
            continue
        located, mentions, n, trimmed, kept = write_fullasr_episodes(con, row, have, index_points_of(row))
        lab = Counter(r["label"] for r in have.values())
        about_min = sum(e["duration_s"] for e in located) / 60
        cut_ok = cut_bad = 0
        cut_s = 0.0
        if not no_cut and located:
            cut_ok, cut_bad, cut_s = cut_meeting(con, row)
        hours = max((w["end_s"] for w in wins), default=0) / 3600
        summary[vid] = {"at": state.now_iso(), "body": row["speaker"], "hours": round(hours, 2), "windows": len(wins), "labels": dict(lab),
                        "episodes": len(located), "mentions": len(mentions), "about_min": round(about_min, 1),
                        "located_min_pretrim": round(sum((e["end_s"] - e["start_s"]) for e in located) / 60 + sum((f[1] - f[0]) - ((p[1] - p[0]) if p else 0) for _, f, p, _ in trimmed) / 60, 1),
                        "trimmed": [{"span": [f[0], f[1]], "piece": list(p) if p else None, "covered_by": c} for _, f, p, c in trimmed],
                        "rows_written": n, "clips_cut": cut_ok, "clips_failed": cut_bad, "clip_min": round(cut_s / 60, 1),
                        "cost_usd": round(sum(r["cost_usd"] or 0 for r in have.values()), 4)}
        con.execute("INSERT INTO events (video_id, from_status, to_status, at, note) VALUES (?,?,?,?,?)",
                    (vid, row["status"], row["status"], state.now_iso(),
                     f"fullasr v2.1: {len(wins)} windows, {len(located)} episodes ({about_min:.0f} about-min), {len(mentions)} mentions, "
                     f"{cut_ok} clips cut, ${summary[vid]['cost_usd']:.4f}"[:300]))
        log(f"derived {vid} ({row['speaker']}): {len(wins)} windows {dict(lab)}; {len(located)} episodes / {about_min:.0f} min "
            f"({len(trimmed)} trimmed/dropped against transcribed clips), {len(mentions)} mentions; {cut_ok} clips cut, {cut_bad} failed")
    jdump(OUT / "derived.json", summary)


# --- 6. the diarized episode batch -------------------------------------------------------------

def propose(con, min_hours=8.0, dry_run=False):
    bodies = sorted({m["speaker"] for m in meetings(con)})
    ready = con.execute(f"SELECT COUNT(*), COALESCE(SUM(transcribed_s), 0) FROM videos WHERE kind = 'episode' AND in_frame = 1 AND status = 'transcoded' "
                        f"AND speaker IN ({','.join('?' * len(bodies))})", bodies).fetchone()
    h = ready[1] / 3600
    pods = gpu_batch.active_pods(con)
    log(f"{ready[0]} clips / {h:.2f} h transcoded for {', '.join(bodies)}; live pods: {', '.join(p['pod_id'] for p in pods) or 'none'}")
    if h < min_hours:
        log(f"below {min_hours} h; no batch proposed")
        return None
    if pods:
        log("a pod is live; wait")
        return None
    if dry_run:
        return None
    return gpu_batch.propose(con, min_hours=min_hours, quiet=True, speakers=bodies)


# --- 7. report -------------------------------------------------------------------------------

V1_METHODS = ("classifier", "local_asr", "agenda_index", "swagit_transcript")


def report(con):
    sc = scope(con, quiet=True)
    bodies = sorted({m["body"] for m in sc})
    L = ["| body | meetings | hours | audio on disk | transcribed (GPU) | excluded | v2.1 read | v2.1 located (h, merged windows, pre-trim) | v2.1 episode rows (h: new cuts + covered) | v1 episodes (h, all v1 routes) | v1 transcribed (h) |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    derived = jload(OUT / "derived.json", {})
    tot = Counter()
    for b in bodies:
        ms = [m for m in sc if m["body"] == b]
        vids = [m["video_id"] for m in ms]
        q = ",".join("?" * len(vids))
        v21 = con.execute(f"SELECT COUNT(*), COALESCE(SUM(core_end_s - core_start_s), 0) FROM episodes WHERE video_id IN ({q}) AND method = ? AND status != 'mention'", vids + [METHOD]).fetchone()
        v1 = con.execute(f"SELECT COUNT(*), COALESCE(SUM(core_end_s - core_start_s), 0) FROM episodes WHERE video_id IN ({q}) AND method IN ({','.join('?' * len(V1_METHODS))}) AND status != 'mention'", vids + list(V1_METHODS)).fetchone()
        v1d = con.execute(f"SELECT COUNT(*), COALESCE(SUM(end_s - start_s), 0) FROM episodes WHERE video_id IN ({q}) AND method IN ({','.join('?' * len(V1_METHODS))}) AND status = 'done'", vids + list(V1_METHODS)).fetchone()
        read = [m for m in ms if m["video_id"] in derived and "episodes" in derived[m["video_id"]]]
        pre_h = sum(derived[m["video_id"]].get("located_min_pretrim", 0) for m in read) / 60
        pre_n = sum(derived[m["video_id"]].get("episodes", 0) + len([t for t in derived[m["video_id"]].get("trimmed", []) if t.get("piece") is None]) for m in read)
        r = {"n": len(ms), "h": sum(m["hours"] or 0 for m in ms), "audio": sum(1 for m in ms if m["audio"]), "tr": sum(1 for m in ms if m["transcript"]), "pre_h": pre_h, "pre_n": pre_n,
             "trh": sum((m["hours"] or 0) for m in ms if m["transcript"]), "ex": sum(1 for m in ms if m["excluded"]), "read": len(read),
             "readh": sum(m["hours"] or 0 for m in read), "v21n": v21[0], "v21h": v21[1] / 3600, "v1n": v1[0], "v1h": v1[1] / 3600, "v1dn": v1d[0], "v1dh": v1d[1] / 3600}
        tot.update(r)
        L.append(f"| {b} | {r['n']} | {r['h']:.1f} | {r['audio']} | {r['tr']} ({r['trh']:.1f} h) | {r['ex']} | {r['read']} ({r['readh']:.1f} h) | "
                 f"{r['pre_n']} ({r['pre_h']:.2f}) | {r['v21n']} ({r['v21h']:.2f}) | {r['v1n']} ({r['v1h']:.2f}) | {r['v1dn']} ({r['v1dh']:.2f}) |")
    L.append(f"| **total** | {tot['n']} | {tot['h']:.1f} | {tot['audio']} | {tot['tr']} ({tot['trh']:.1f} h) | {tot['ex']} | {tot['read']} ({tot['readh']:.1f} h) | "
             f"{tot['pre_n']} ({tot['pre_h']:.2f}) | {tot['v21n']} ({tot['v21h']:.2f}) | {tot['v1n']} ({tot['v1h']:.2f}) | {tot['v1dn']} ({tot['v1dh']:.2f}) |")
    calls = con.execute("SELECT COUNT(*), COALESCE(SUM(cost_usd), 0), COALESCE(SUM(input_tokens), 0), COALESCE(SUM(cache_read_input_tokens), 0), "
                        "COALESCE(SUM(cache_creation_input_tokens), 0), COALESCE(SUM(output_tokens), 0) FROM classifier_calls WHERE video_id LIKE ? AND prompt_version = ?",
                        (CALL_PREFIX + "%", pv())).fetchone()
    labels = dict(Counter(r[0] for r in con.execute("SELECT label FROM classifier_calls WHERE video_id LIKE ? AND prompt_version = ? AND label IS NOT NULL", (CALL_PREFIX + "%", pv()))))
    g = jload(GPU_LOG, {})
    L += ["", f"Sonnet (fullasr v2.1, prompt_version `{pv()}`): {calls[0]} calls, **${calls[1]:.4f}**; tokens in {calls[2]:,} / cache-read {calls[3]:,} / "
              f"cache-write {calls[4]:,} / out {calls[5]:,}; labels {labels}; Claude total in DB ${spend.claude_total(con):.2f} of cap ${spend.caps()['CLAUDE_CAP_USD']:.0f}",
          "", "| GPU batch | meetings | audio-h | est (RTFX 25x) | est at measured $0.0102/h | actual | GPU-min | status | loaded / rescued (dead air) / QA FAIL |", "|---|---|---|---|---|---|---|---|---|"]
    for bid, v in sorted(g.items()):
        resc = {k: x for k, x in (v.get("rescued") or {}).items() if not x.get("dry_run")}
        L.append(f"| {bid} | {len(v.get('meetings', []))} | {v.get('hours')} | ${v.get('est_usd')} | ${v.get('est_measured_rate_usd', '-')} | "
                 f"{('$%.4f' % v['actual_usd']) if v.get('actual_usd') is not None else '-'} | {v.get('gpu_minutes', '-')} | {v.get('status', 'reused' if v.get('reused') else '?')} | "
                 f"{len(v.get('harvested', v.get('meetings', [])))} / {len(resc)} / {len(v.get('failed') or {})}"
                 f"{(' (' + ', '.join(sorted(v['failed'])) + ')') if v.get('failed') else ''} |")
    L += ["", " ".join(spend.lines(spend.summary(con)))]
    text = "\n".join(L)
    (OUT / "report.md").write_text(text + "\n")
    print(text)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["scope", "fetch", "encode", "reuse", "prepare", "harvest", "rescue", "windows", "submit", "collect", "derive", "propose", "report"])
    ap.add_argument("id", nargs="?")
    ap.add_argument("--bodies", default=None)
    ap.add_argument("--ids", nargs="+", default=None)
    ap.add_argument("--hours", type=float, default=95.0, help="prepare: audio-hours per GPU batch")
    ap.add_argument("--max-hours", type=float, default=None, help="fetch: stop after this many audio-hours")
    ap.add_argument("--max", type=int, default=0)
    ap.add_argument("--max-windows", type=int, default=0)
    ap.add_argument("--cap-usd", type=float, default=15.0)
    ap.add_argument("--min-hours", type=float, default=8.0)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-cut", action="store_true")
    ap.add_argument("--no-derive", action="store_true")
    ap.add_argument("--state", default=None)
    a = ap.parse_args(argv)
    for d in (OUT, AUDIO, OPUS, WHISPER, BATCHES, WINDOWS, CLIPS, LOGS):
        d.mkdir(parents=True, exist_ok=True)
    bodies = [b.strip() for b in a.bodies.split(",") if b.strip()] if a.bodies else None
    con = state.connect(a.state)
    if a.cmd == "scope":
        scope(con, bodies); return 0
    if a.cmd == "fetch":
        return fetch(con, bodies, a.max_hours, a.max)
    if a.cmd == "encode":
        encode(con, bodies); return 0
    if a.cmd == "reuse":
        reuse(con); return 0
    if a.cmd == "prepare":
        prepare(con, a.hours, bodies, a.dry_run); return 0
    if a.cmd == "harvest":
        if not a.id:
            ap.error("harvest needs a batch id")
        harvest(con, a.id); return 0
    if a.cmd == "rescue":
        if not a.id:
            ap.error("rescue needs a batch id")
        rescue(con, a.id, a.dry_run); return 0
    if a.cmd == "windows":
        build_windows(a.ids); return 0
    if a.cmd == "submit":
        return submit(con, a.cap_usd, bodies, a.max_windows, a.dry_run)
    if a.cmd == "collect":
        return collect(con, a.no_derive)
    if a.cmd == "derive":
        derive(con, a.ids, a.no_cut); return 0
    if a.cmd == "propose":
        propose(con, a.min_hours, a.dry_run); return 0
    report(con)
    return 0


if __name__ == "__main__":
    sys.exit(main())
