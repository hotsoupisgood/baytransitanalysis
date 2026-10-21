#!/usr/bin/env python3
"""
scripts/recall_check_yt.py -- the locate-recall check (scripts/recall_check.py,
plan/RECALL-CHECK.md) on the YouTube AUTO-CAPTION arm (2026-10-18).

Same method as the Richmond / San Leandro check, on YouTube meetings whose
locate track is the YouTube auto-caption VTT (`videos.captions = 'auto'`,
`episodes.method = 'classifier'`) and whose whole-meeting opus is already on
disk (`output/audio/<id>.opus`; nothing is fetched from YouTube). Everything
is written under output/validation/recall_yt/; the synthetic `videos` rows are
`rcyt-<id>` (kind='validation', in_frame=0), the classifier rows are
`video_id='recallyt:<id>'`, the batch note carries 'validation-yt'.

    sample --ids A,B,C,D        record the sample (chosen by hand; see plan/RECALL-CHECK.md)
    prepare                     copy opus -> audio/; videos rows; batch row (then approve + nohup launch)
    tag-spend / harvest / windows / submit / collect / analyze   as recall_check.py
    quality                     caption words per audio-h vs WhisperX; WER of the captions vs WhisperX
    recompute                   recall under the old (git 35df7c7) and the current keyword list -> kw_old/, kw_new/
"""
import argparse, json, re, shutil, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))
sys.path.insert(0, str(ROOT / "scripts"))
import state                    # noqa: E402
import segment as sg            # noqa: E402
import classify as cl           # noqa: E402
import gpu_batch                # noqa: E402
import recall_check as rc       # noqa: E402
import recall_recompute as rr   # noqa: E402

OUT = ROOT / "output" / "validation" / "recall_yt"
AUDIO_DIR = ROOT / "output" / "audio"
OLD_KEYWORDS_REV = "35df7c7"            # the English pair as committed = kw-dfd392a1fe

# --- re-point recall_check at this arm -----------------------------------------------
rc.OUT = OUT
rc.METHOD = "classifier"
rc.CAPTIONS_KIND = "auto"
rc.PREFIX = "rcyt-"
rc.CALL_PREFIX = "recallyt:"
rc.API_ID_PREFIX = "recallyt-"
rc.NOTE = "validation: locate-recall check, YouTube auto-caption arm [validation-yt] (plan/RECALL-CHECK.md)"
rc.transcode = lambda src, dst: (shutil.copy2(src, dst), 0.0)[1]     # the opus IS the corpus transcode already; copy, follow symlinks

log = rc.log


def sample(con, ids, why):
    rows = []
    for vid in ids:
        r = con.execute("SELECT video_id, speaker, meeting_date, duration_s, status, captions, captions_path, opus_path FROM videos "
                        "WHERE video_id = ?", (vid,)).fetchone()
        if r is None:
            sys.exit(f"{vid}: no videos row")
        if r["captions"] != "auto":
            sys.exit(f"{vid}: captions = {r['captions']!r}, not 'auto'")
        opus = AUDIO_DIR / f"{vid}.opus"
        if not opus.exists():
            sys.exit(f"{vid}: {opus} missing")
        if not r["captions_path"] or not Path(r["captions_path"]).exists():
            sys.exit(f"{vid}: caption track missing ({r['captions_path']})")
        dur = rc.duration(opus)
        if abs(dur - (r["duration_s"] or 0)) > 120:
            sys.exit(f"{vid}: opus is {dur:.0f} s but the meeting is {r['duration_s']} s -- not whole-meeting audio")
        eps = con.execute("SELECT COUNT(*) n, COALESCE(SUM(end_s - start_s), 0) s, GROUP_CONCAT(DISTINCT tags) t FROM episodes "
                          "WHERE video_id = ? AND method = ? AND status != 'mention'", (vid, rc.METHOD)).fetchone()
        cap = sg.read_segments(r["captions_path"])
        rows.append({"video_id": vid, "body": r["speaker"], "date": r["meeting_date"], "hours": round(dur / 3600, 2),
                     "status": r["status"], "wav": str(opus.resolve()), "opus_src": str(opus), "vtt": r["captions_path"],
                     "episodes": eps["n"], "episode_min": round(eps["s"] / 60, 1), "tags": eps["t"],
                     "caption_words": sum(len(x["text"].split()) for x in cap), "caption_segments": len(cap)})
        log(f"  {vid} {r['speaker']:10} {r['meeting_date']} {dur / 3600:5.2f} h  {eps['n']:2d} episodes {eps['s'] / 60:6.1f} min  "
            f"{eps['t'] or '-'}  captions {rows[-1]['caption_words']:,} w")
    OUT.mkdir(parents=True, exist_ok=True)
    doc = {"arm": "youtube_auto_captions", "rule": why, "hours": round(sum(r["hours"] for r in rows), 2), "meetings": rows,
           "at": state.now_iso()}
    rc.sample_path().write_text(json.dumps(doc, indent=1))
    log(f"sample: {len(rows)} meetings, {doc['hours']:.2f} audio-h -> {rc.sample_path()}")


def prepare(con):
    rc.prepare(con)
    s = rc.load_sample()
    # the estimate recall_check prints assumes diarization (RTFX 25); the measured no-diarization
    # rate on b35 was $0.0102 per audio-h plus boot -- report both
    c = gpu_batch.cfg()
    h = s["audio_hours"]
    s["gpu_est_measured_usd"] = round(h * 0.0102 + c["boot_min"] / 60 * c["usd_hr"], 3)
    rc.sample_path().write_text(json.dumps(s, indent=1))
    log(f"GPU: cfg estimate ${s['gpu_est_usd']:.2f} (RTFX {c['rtfx']:.0f}x, x{c['contingency']:.1f}, diarization assumed); "
        f"measured-rate estimate ${s['gpu_est_measured_usd']:.3f} ({h:.2f} h x $0.0102/h + {c['boot_min']:.0f} min boot at ${c['usd_hr']:.2f}/h)")


# --- caption quality: density and WER against the WhisperX text --------------------------

_BRACKET = re.compile(r"\[[^\]]*\]")
_PUNCT = re.compile(r"[^\w\s']")


def norm_words(text):
    t = _BRACKET.sub(" ", text.lower().replace(">>", " "))
    t = t.replace("-", " ").replace("’", "'")
    t = _PUNCT.sub(" ", t)
    return [w.strip("'") for w in t.split() if w.strip("'")]


def wer(ref, hyp):
    from rapidfuzz.distance import Levenshtein
    if not ref:
        return None
    return Levenshtein.distance(ref, hyp) / len(ref)


def quality(con):
    s = rc.load_sample()
    out = {"at": state.now_iso(), "reference": "GPU WhisperX large-v3 whole-meeting text (same config as the corpus)",
           "hypothesis": "YouTube auto-caption VTT as parsed by segment.vtt_to_segments (the locate track)",
           "normalisation": "lower-case, [bracket] tags and >> removed, hyphens split, punctuation stripped", "meetings": []}
    chunk = 600.0
    for m in s["meetings"]:
        if not m.get("transcript"):
            continue
        doc = json.loads(Path(m["transcript"]).read_text())
        gseg = [{"start": float(x["start"]), "end": float(x["end"]), "text": x["text"]} for x in doc["segments"]]
        cseg = sg.read_segments(m["vtt"])
        hours = m["audio_s"] / 3600
        gw = sum(len(x["text"].split()) for x in gseg)
        cw = sum(len(x["text"].split()) for x in cseg)
        ref = norm_words(" ".join(x["text"] for x in gseg))
        hyp = norm_words(" ".join(x["text"] for x in cseg))
        w = wer(ref, hyp)
        # per 10-min chunk, by timestamp, for the spread
        end = max(max((x["end"] for x in gseg), default=0), max((x["end"] for x in cseg), default=0))
        chunks = []
        t = 0.0
        while t < end:
            r_ = norm_words(cl.window_text(gseg, t, t + chunk))
            h_ = norm_words(cl.window_text(cseg, t, t + chunk))
            if len(r_) >= 50:
                chunks.append(round(wer(r_, h_), 3))
            t += chunk
        chunks.sort()
        q = lambda p: chunks[min(len(chunks) - 1, int(p * len(chunks)))] if chunks else None
        row = {"video_id": m["video_id"], "body": m["body"], "hours": round(hours, 2),
               "whisperx_words": gw, "caption_words": cw, "whisperx_words_per_h": round(gw / hours), "caption_words_per_h": round(cw / hours),
               "caption_to_whisperx_word_ratio": round(cw / gw, 3) if gw else None,
               "wer": round(w, 4) if w is not None else None, "wer_chunks_10min": {"n": len(chunks), "median": q(0.5), "p90": q(0.9), "max": chunks[-1] if chunks else None},
               "gpu_qa": m.get("ingest_status"), "gpu_qa_note": m.get("ingest_note")}
        out["meetings"].append(row)
        log(f"  {m['video_id']} {m['body']:10} {hours:.2f} h  WhisperX {gw:,} w ({row['whisperx_words_per_h']:,}/h)  captions {cw:,} w "
            f"({row['caption_words_per_h']:,}/h, x{row['caption_to_whisperx_word_ratio']})  WER {w:.3f}  10-min median {q(0.5)} p90 {q(0.9)}")
    (OUT / "caption_quality.json").write_text(json.dumps(out, indent=1))
    lines = ["| meeting | body | h | WhisperX words/h | caption words/h | ratio | WER (captions vs WhisperX) | 10-min median | p90 |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in out["meetings"]:
        lines.append(f"| {r['video_id']} | {r['body']} | {r['hours']:.2f} | {r['whisperx_words_per_h']:,} | {r['caption_words_per_h']:,} | "
                     f"{r['caption_to_whisperx_word_ratio']:.2f} | {r['wer']:.3f} | {r['wer_chunks_10min']['median']} | {r['wer_chunks_10min']['p90']} |")
    (OUT / "caption_quality.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def recompute(con, out_new="kw_new"):
    """kw_old/ = the list as committed at OLD_KEYWORDS_REV; <out_new>/ = the
    list on disk WITH the auto-caption alias layer (classify.keywords_for(
    'en', 'auto'), 2026-10-18) -- this arm's locate tracks are captions='auto',
    so that is the list the pipeline now screens them with. Pipeline labels
    are bounds-matched at the 240-s granularity the rows were read at."""
    old = rr.keywords_at(OLD_KEYWORDS_REV)
    new = cl.keywords_for("en", "auto")
    log(f"old list {old.version} ({len(old)} terms) -> {OUT / 'kw_old'}; current list {new.version} ({len(new)} terms) -> {OUT / out_new}")
    rr.recompute(con, OUT, OUT / "kw_old", old, cl.PV_MAX_RUN)
    rr.recompute(con, OUT, OUT / out_new, new, cl.PV_MAX_RUN)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["sample", "prepare", "tag-spend", "harvest", "windows", "submit", "collect", "analyze", "quality", "recompute"])
    ap.add_argument("id", nargs="?")
    ap.add_argument("--ids", default=None)
    ap.add_argument("--why", default="")
    ap.add_argument("--cap-usd", type=float, default=3.0)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out-new", default="kw_new", help="recompute: subdirectory for the current-list arm (default kw_new)")
    ap.add_argument("--state", default=None)
    a = ap.parse_args(argv)
    con = state.connect(a.state)
    OUT.mkdir(parents=True, exist_ok=True)
    if a.cmd == "sample":
        sample(con, [x.strip() for x in (a.ids or "").split(",") if x.strip()], a.why)
    elif a.cmd == "prepare":
        prepare(con)
    elif a.cmd == "tag-spend":
        rc.tag_spend(con, a.id)
    elif a.cmd == "harvest":
        rc.harvest(con, a.id)
    elif a.cmd == "windows":
        rc.build_windows(con)
        s = rc.load_sample()
        s["keywords_version"] = cl.keywords_for("en").version
        rc.sample_path().write_text(json.dumps(s, indent=1))
    elif a.cmd == "submit":
        return rc.submit(con, a.cap_usd, a.dry_run)
    elif a.cmd == "collect":
        return rc.collect(con)
    elif a.cmd == "analyze":
        return rc.analyze(con)
    elif a.cmd == "quality":
        quality(con)
    elif a.cmd == "recompute":
        recompute(con, a.out_new)
    return 0


if __name__ == "__main__":
    sys.exit(main())
