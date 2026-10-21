#!/usr/bin/env python3
"""
hasan_compare.py
================
Pull recent HasanAbi videos in a target length window, grab the official
YouTube captions, transcribe the same audio with Whisper medium, and emit
a side-by-side HTML diff per video (bucketed by timestamp).

Fixed 2026-09-17 (see docs/REVIEW-2026-09-17.md):
  * WER now divides by the CAPTION (reference) word count. It used Whisper as
    the reference, which halves the error a hallucinated passage adds.
  * Text is normalised with analysis/qa_transcripts.norm_words: YouTube's VTT
    writes profanity as "[&nbsp;__&nbsp;]", which the old normaliser turned into
    three error tokens per swear word (Hasan's videos read 2-3x worse).
  * Caption kind (manual vs YouTube's own ASR) is read from the file content.
    Every diff page so far said "manual"; all 20 were auto-generated.
  * Subtitles are requested as en/en-orig/en-US/en-GB, not "en.*", which also
    matched ~100 machine-translated tracks (en-hi, en-nl...) and got HTTP 429.
  * One failing video no longer aborts the whole run.

Quick start:
    pip3 install --user yt-dlp faster-whisper webvtt-py
    # ffmpeg must be on PATH (brew install ffmpeg)
    python3 legacy/hasan_compare.py --n 3 --min-min 10 --max-min 30

Output (next to this file):
    output/audio/<id>.m4a
    output/subs/<id>.<lang>.vtt        # raw, from YouTube
    output/subs/<id>.txt                # cleaned plain text
    output/whisper/<id>.json            # whisper segments w/ timestamps
    output/whisper/<id>.txt             # cleaned plain text
    output/diffs/<id>.html              # side-by-side HTML
    output/diffs/index.html             # links to all diffs
"""

import argparse
import html
import json
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Lives in legacy/ since 2026-10-16; the repo root is one level up.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "analysis"))
from qa_transcripts import edit_distance, norm_words, INLINE_TS   # noqa: E402

DEFAULT_CHANNEL = "https://www.youtube.com/@hasanabi/videos"
OUT = ROOT / "output"
AUDIO_DIR = OUT / "audio"
SUBS_DIR = OUT / "subs"
WHISPER_DIR = OUT / "whisper"
DIFF_DIR = OUT / "diffs"


# ---------- helpers ----------------------------------------------------------

def run(cmd, **kw):
    print("$", " ".join(str(c) for c in cmd))
    return subprocess.run(cmd, check=True, **kw)


def fmt_time(sec: float) -> str:
    sec = int(sec)
    h, m, s = sec // 3600, (sec % 3600) // 60, sec % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


# ---------- step 1: pick videos ---------------------------------------------

def list_candidates(pool: int = 60, channel: str = DEFAULT_CHANNEL):
    """Return [{'id','title','duration'}, ...] for the channel's most recent uploads."""
    cmd = [
        "yt-dlp", "--flat-playlist",
        "--print", "%(id)s\t%(title)s\t%(duration)s",
        "--playlist-items", f"1-{pool}",
        channel,
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    out = []
    for line in res.stdout.strip().splitlines():
        parts = line.split("\t", 2)
        if len(parts) != 3:
            continue
        vid, title, dur = parts
        try:
            dur = int(float(dur)) if dur and dur != "NA" else 0
        except ValueError:
            dur = 0
        out.append({"id": vid, "title": title, "duration": dur})
    return out


def pick_videos(candidates, n, min_s, max_s, skip_ids=None):
    skip = set(skip_ids or [])
    fits = [c for c in candidates if min_s <= c["duration"] <= max_s and c["id"] not in skip]
    return fits[:n]


# ---------- step 2: download audio + YT subs --------------------------------

def download(video):
    url = f"https://www.youtube.com/watch?v={video['id']}"
    audio_path = AUDIO_DIR / f"{video['id']}.m4a"

    # Subtitles first (fast, doesn't pull media)
    run([
        "yt-dlp",
        "--skip-download",
        "--write-auto-subs", "--write-subs",
        # Not "en.*": that regex also matches every machine translation INTO
        # English (en-hi, en-nl, ...), ~100 extra requests per video -> HTTP 429.
        "--sub-langs", "en,en-orig,en-US,en-GB",
        "--sub-format", "vtt",
        "-o", str(SUBS_DIR / "%(id)s.%(ext)s"),
        url,
    ])

    # Audio (m4a, no re-encoding)
    if not audio_path.exists():
        run([
            "yt-dlp",
            "-f", "bestaudio[ext=m4a]/bestaudio",
            "-o", str(AUDIO_DIR / "%(id)s.%(ext)s"),
            "--no-playlist",
            url,
        ])
    return audio_path


# ---------- step 3: parse YT VTT --------------------------------------------

def vtt_to_segments(vtt_path: Path):
    """Parse a YT caption VTT file into timed segments.

    Auto-generated captions use a rolling two-line display where every cue
    repeats the previous phrase on line 1 and adds new words (with embedded
    <HH:MM:SS.mmm> timing markers) on line 2.  Extracting only the line that
    carries timing markers gives clean, non-duplicated text.

    Separator cues (no timing markers, just the completed phrase) are skipped.

    Manually-uploaded captions have no inline timing markers; they are parsed
    straight — all cue text is taken as-is.
    """
    raw = vtt_path.read_text(encoding="utf-8")
    is_auto = bool(re.search(r"<\d{2}:\d{2}:\d{2}", raw))

    def _ts(s):
        s = s.strip().split()[0].replace(",", ".")
        h, m, sec = s.split(":")
        return int(h) * 3600 + int(m) * 60 + float(sec)

    segs = []
    for block in re.split(r"\n{2,}", raw):
        lines = [l for l in block.split("\n") if l.strip()]
        if not lines:
            continue
        ts_idx = next((i for i, l in enumerate(lines) if "-->" in l), None)
        if ts_idx is None:
            continue
        ts = lines[ts_idx]
        content = lines[ts_idx + 1:]
        if not content:
            continue

        if is_auto:
            # Only take the line that carries new inline timing markers.
            # Cues without them are separator snapshots — skip.
            new_line = next(
                (l for l in content if re.search(r"<\d{2}:\d{2}:\d{2}", l)), None
            )
            if new_line is None:
                continue
            text = re.sub(r"<[^>]+>", "", new_line)
        else:
            text = " ".join(re.sub(r"<[^>]+>", "", l) for l in content)

        text = re.sub(r"\s+", " ", html.unescape(text)).strip()
        if not text:
            continue

        try:
            parts = ts.split("-->")
            start, end = _ts(parts[0]), _ts(parts[1])
        except Exception:
            continue

        segs.append({"start": start, "end": end, "text": text})
    return segs


def pick_yt_vtt(video_id: str):
    """Prefer manual English captions over auto-generated.

    yt-dlp names both kinds <id>.en.vtt, so the old filename test ("auto" not in
    the name) labelled every auto-caption file "manual". The content decides:
    YouTube's ASR captions carry inline <hh:mm:ss.mmm> word timings."""
    candidates = sorted(SUBS_DIR.glob(f"{video_id}.en*.vtt"))
    manual, auto = [], []
    for p in candidates:
        (auto if INLINE_TS.search(p.read_text(encoding="utf-8")) else manual).append(p)
    if manual:
        return manual[0], "manual"
    if auto:
        return auto[0], "auto"
    return None, None


# ---------- step 4: whisper -------------------------------------------------

def whisper_segments(audio_path: Path, model_name: str = "medium"):
    """Run faster-whisper on audio_path and return timed segments + elapsed seconds."""
    import time
    from faster_whisper import WhisperModel
    model = WhisperModel(model_name, device="cpu", compute_type="int8")
    t0 = time.time()
    segs, _info = model.transcribe(
        str(audio_path),
        beam_size=5,
        vad_filter=True,
        language="en",
    )
    result = [
        {"start": s.start, "end": s.end, "text": s.text.strip()}
        for s in segs
    ]
    elapsed = time.time() - t0
    return result, elapsed


# ---------- step 5: WER -------------------------------------------------------

def _normalize(text: str) -> list[str]:
    """Word list for WER: see analysis/qa_transcripts.norm_words."""
    return norm_words(text)


def wer(ref: str, hyp: str) -> float:
    """Word Error Rate = (S+D+I) / len(REFERENCE words).

    ref = the captions (or the larger model), hyp = the transcript being judged.
    Order matters: with Whisper as the reference, a hallucinated passage grows
    the denominator and LOWERS the reported error. Uses the bit-parallel edit
    distance, so a 3-hour stream takes seconds instead of hours."""
    r = _normalize(ref)
    h = _normalize(hyp)
    if not r:
        return 0.0
    return edit_distance(r, h) / len(r)


# ---------- step 6: bucketed side-by-side -----------------------------------

def bucketize(segs, bucket: float = 30.0):
    buckets = {}
    for s in segs:
        b = int(s["start"] // bucket)
        buckets.setdefault(b, []).append(s["text"])
    return {b: " ".join(t.strip() for t in texts) for b, texts in buckets.items()}


def render_html(video, yt_buckets, ws_buckets, kind, out_path: Path,
                bucket: float = 30.0, model_name: str = "medium"):
    keys = sorted(set(yt_buckets) | set(ws_buckets))

    # Overall WER across all buckets (concatenate in time order)
    all_ws = " ".join(ws_buckets.get(k, "") for k in keys)
    all_yt = " ".join(yt_buckets.get(k, "") for k in keys)
    overall = wer(all_yt, all_ws)          # reference = captions

    rows = []
    for k in keys:
        t = fmt_time(k * bucket)
        yt_raw = yt_buckets.get(k, "")
        ws_raw = ws_buckets.get(k, "")
        yt = html.escape(yt_raw)
        ws = html.escape(ws_raw)
        if yt_raw and ws_raw:
            w = wer(yt_raw, ws_raw)
            pct = f"{w * 100:.0f}%"
            # green <10%, yellow <30%, red >=30%
            wer_cls = "wer-lo" if w < 0.10 else ("wer-mid" if w < 0.30 else "wer-hi")
            wer_cell = f'<td class="wer {wer_cls}">{pct}</td>'
        else:
            wer_cell = '<td class="wer">—</td>'
        rows.append(
            f'<tr><td class="t">{t}</td>{wer_cell}'
            f'<td>{yt}</td><td>{ws}</td></tr>'
        )

    title = html.escape(video["title"])
    body = "\n".join(rows)
    doc = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>{title}</title>
<style>
 body{{font-family:-apple-system,Segoe UI,sans-serif;margin:0;padding:1rem;background:#fafafa;color:#111}}
 h1{{font-size:1.05rem;margin:0 0 .25rem}}
 .meta{{color:#666;font-size:.82rem;margin-bottom:1rem}}
 .overall{{display:inline-block;margin-left:.75rem;font-weight:600;color:#333}}
 table{{border-collapse:collapse;width:100%;table-layout:fixed}}
 th,td{{vertical-align:top;padding:.45rem .6rem;border-bottom:1px solid #eee;font-size:.9rem;line-height:1.35}}
 th{{position:sticky;top:0;background:#fff;text-align:left;border-bottom:2px solid #ccc}}
 td.t{{width:72px;color:#888;font-family:ui-monospace,monospace;font-size:.78rem}}
 td.wer{{width:52px;text-align:right;font-family:ui-monospace,monospace;font-size:.78rem;font-weight:600}}
 .wer-lo{{color:#2a9d2a}}
 .wer-mid{{color:#c97d00}}
 .wer-hi{{color:#cc2200}}
 td:nth-child(3){{border-right:1px solid #eee}}
 td:nth-child(3),td:nth-child(4){{word-wrap:break-word}}
</style></head>
<body>
<h1>{title}</h1>
<div class="meta">
 YouTube ID {video['id']} • {video['duration']}s • bucket {int(bucket)}s • YT source: {kind}{" (YouTube's own ASR -- agreement, not accuracy)" if kind == "auto" else ""}
 <span class="overall">WER of Whisper vs captions: {overall * 100:.1f}%</span>
</div>
<table>
 <thead><tr><th>Time</th><th>WER</th><th>YouTube captions ({kind})</th><th>Whisper {html.escape(model_name)}</th></tr></thead>
 <tbody>
{body}
 </tbody></table>
</body></html>"""
    out_path.write_text(doc, encoding="utf-8")
    return overall


def write_index():
    pages = sorted(p for p in DIFF_DIR.glob("*.html") if p.name != "index.html")
    items = "\n".join(
        f'<li><a href="{p.name}">{html.escape(p.stem)}</a></li>' for p in pages
    )
    (DIFF_DIR / "index.html").write_text(
        f"""<!doctype html><html><head><meta charset="utf-8">
<title>YouTube captions vs Whisper</title>
<style>body{{font-family:-apple-system,sans-serif;padding:1rem}}</style></head>
<body><h1>YouTube captions vs Whisper</h1><ul>{items}</ul>
</body></html>""",
        encoding="utf-8",
    )


# ---------- parallel helpers ------------------------------------------------

def _download_one(video):
    """Download audio + subs for one video (runs in a thread). A failure is
    returned, not raised: one bad video must not abort the batch (the Tucker run
    of 2026-05-04 lost all five videos to one subtitle 429)."""
    try:
        audio = download(video)
    except subprocess.CalledProcessError as e:
        return video, None, None, f"yt-dlp exit {e.returncode}"
    vtt_path, kind = pick_yt_vtt(video["id"])
    return video, audio, vtt_path, kind


def _transcribe_one(video, audio_path, model_name):
    """Transcribe audio with one model, using cache if available (runs in a thread).
    Returns (segs, elapsed_or_None).
    """
    ws_json = WHISPER_DIR / f"{video['id']}.{model_name}.json"
    # Back-compat: rename legacy <id>.json → <id>.medium.json
    legacy = WHISPER_DIR / f"{video['id']}.json"
    if model_name == "medium" and legacy.exists() and not ws_json.exists():
        legacy.rename(ws_json)
    if ws_json.exists():
        return json.loads(ws_json.read_text(encoding="utf-8")), None
    segs, elapsed = whisper_segments(audio_path, model_name)
    ws_json.write_text(json.dumps(segs, indent=2), encoding="utf-8")
    (WHISPER_DIR / f"{video['id']}.{model_name}.txt").write_text(
        "\n".join(s["text"] for s in segs), encoding="utf-8"
    )
    return segs, elapsed


# ---------- main ------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3, help="how many videos")
    ap.add_argument("--min-min", type=int, default=10, help="min length (minutes)")
    ap.add_argument("--max-min", type=int, default=30, help="max length (minutes)")
    ap.add_argument("--bucket", type=int, default=30, help="diff bucket size (seconds)")
    ap.add_argument("--pool", type=int, default=60, help="how many recent videos to consider")
    ap.add_argument("--models", default="medium",
                    help="comma-separated whisper model names, e.g. tiny,small,medium")
    ap.add_argument("--workers", type=int, default=2,
                    help="parallel transcription workers (default 2; each loads a full model)")
    ap.add_argument("--channel", default=DEFAULT_CHANNEL,
                    help="YouTube channel URL to pull videos from")
    ap.add_argument("--speaker", default=None,
                    help="Speaker label (e.g. HasanAbi). Inferred from channel URL if omitted.")
    args = ap.parse_args()
    model_names = [m.strip() for m in args.models.split(",") if m.strip()]

    speaker = args.speaker or re.sub(r"[/@]", "", args.channel.split("@")[-1].split("/")[0])

    for d in (AUDIO_DIR, SUBS_DIR, WHISPER_DIR, DIFF_DIR):
        d.mkdir(parents=True, exist_ok=True)

    speakers_path = OUT / "speakers.json"
    speakers = json.loads(speakers_path.read_text()) if speakers_path.exists() else {}

    print("[1/4] listing candidates...")
    candidates = list_candidates(args.pool, args.channel)
    chosen = pick_videos(candidates, args.n, args.min_min * 60, args.max_min * 60,
                         skip_ids=speakers.keys())
    if len(chosen) < args.n:
        print(f"!! only {len(chosen)} videos fit {args.min_min}-{args.max_min} min — "
              f"try widening with --pool 100 or relaxing length range")
    if not chosen:
        sys.exit(1)
    for v in chosen:
        print(f"   {v['id']}  {v['duration']:>5}s  {v['title'][:80]}")

    for v in chosen:
        speakers[v["id"]] = speaker
    speakers_path.write_text(json.dumps(speakers, indent=2))


    # Phase 1: download all videos in parallel (I/O bound)
    print(f"\n[2/4] downloading {len(chosen)} videos in parallel...")
    downloaded = []
    with ThreadPoolExecutor(max_workers=len(chosen)) as ex:
        futs = {ex.submit(_download_one, v): v for v in chosen}
        for fut in as_completed(futs):
            v, audio, vtt_path, kind = fut.result()
            if audio is None:
                print(f"   [{v['id']}] download FAILED ({kind}) — skipping")
                continue
            if not vtt_path:
                print(f"   [{v['id']}] no captions — skipping")
                continue
            yt_segs = vtt_to_segments(vtt_path)
            (SUBS_DIR / f"{v['id']}.txt").write_text(
                "\n".join(s["text"] for s in yt_segs), encoding="utf-8"
            )
            print(f"   [{v['id']}] download done — {v['title'][:60]}")
            downloaded.append((v, audio, vtt_path, kind, yt_segs))

    # Phase 2: transcribe all (video × model) tasks in parallel (CPU bound, limited workers)
    tasks = [(v, audio, vtt_path, kind, yt_segs, mname)
             for (v, audio, vtt_path, kind, yt_segs) in downloaded
             for mname in model_names]
    print(f"\n[3/4] transcribing {len(tasks)} task(s) with {args.workers} worker(s)...")

    # results keyed by video id
    trans_results: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {
            ex.submit(_transcribe_one, v, audio, mname): (v, audio, vtt_path, kind, yt_segs, mname)
            for (v, audio, vtt_path, kind, yt_segs, mname) in tasks
        }
        for fut in as_completed(futs):
            v, audio, vtt_path, kind, yt_segs, mname = futs[fut]
            try:
                segs, elapsed = fut.result()
            except Exception as e:                      # keep the other videos
                print(f"   [{v['id']}] [{mname}] transcription FAILED: {e}")
                continue
            rtf_str = f"RTF {elapsed/v['duration']:.2f}x" if elapsed and v["duration"] else "cached"
            print(f"   [{v['id']}] [{mname}] done — {rtf_str}")
            rec = trans_results.setdefault(v["id"], {
                "v": v, "vtt_path": vtt_path, "kind": kind,
                "yt_segs": yt_segs, "models": {}, "times": {}
            })
            rec["models"][mname] = segs
            rec["times"][mname] = elapsed

    # Phase 3: render HTML
    print(f"\n[4/4] rendering diffs...")
    ref_order = ["large-v3", "large-v2", "large", "medium", "small", "base", "tiny"]
    for vid_id, rec in trans_results.items():
        v, kind, yt_segs = rec["v"], rec["kind"], rec["yt_segs"]
        model_segs, model_times = rec["models"], rec["times"]

        ref_model = next((m for m in ref_order if m in model_segs), model_names[-1])
        ref_segs = model_segs[ref_model]
        ref_text = " ".join(s["text"] for s in ref_segs)

        yt_b  = bucketize(yt_segs, args.bucket)
        ref_b = bucketize(ref_segs, args.bucket)
        overall_wer = render_html(v, yt_b, ref_b, kind,
                                  DIFF_DIR / f"{vid_id}.html", bucket=args.bucket,
                                  model_name=ref_model)

        print(f"\n   {v['title'][:70]}")
        print(f"   {'Model':<12} {'WER vs '+ref_model:<22} {'RTF':>7}  {'Words':>6}")
        print(f"   {'-'*55}")
        print(f"   {'YT ('+kind+')':<12} {overall_wer*100:>6.1f}%{'':16}  {len(_normalize(ref_text)):>6}")
        for mname in model_names:
            if mname == ref_model:
                print(f"   {mname:<12} {'(reference)':<22}          {len(_normalize(ref_text)):>6}")
                continue
            mtext = " ".join(s["text"] for s in model_segs[mname])
            w = wer(ref_text, mtext)
            t = model_times[mname]
            rtf_str = f"{t/v['duration']:.2f}x" if t and v["duration"] else "cached"
            print(f"   {mname:<12} {w*100:>6.1f}%{'':16} {rtf_str:>7}  {len(_normalize(mtext)):>6}")
        print(f"   -> {DIFF_DIR / (vid_id + '.html')}")

    write_index()
    print(f"\nDone. Open {DIFF_DIR / 'index.html'}")


if __name__ == "__main__":
    main()
