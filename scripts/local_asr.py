#!/usr/bin/env python3
"""
scripts/local_asr.py -- free ASR on this Mac, to LOCATE episodes in meetings
that have neither a usable caption track nor agenda index points
(PROPOSAL-transit.md 2.2 last free row, docs/MODELS.md 4; built 2026-09-22).

WHAT THIS IS FOR, AND WHAT IT IS NOT. The output is a caption-equivalent VTT:
the same input `pipeline/segment.py` reads from a YouTube auto track, so the
existing stage 1 + classifier locate episodes unchanged. It is deliberately
LOW QUALITY -- turbo weights, 16 kHz mono, no alignment, no diarization, no
word timestamps -- because all it has to do is put roughly the right words at
roughly the right second. The located episode is still cut from the real audio
and transcribed on the GPU under the one pinned corpus configuration.
docs/MODELS.md: never mix a locally-transcribed file into the corpus. So the
row it writes is `captions = 'asr_local'`, never 'auto' / 'manual' /
'granicus', and `raw_meta.asr` records the model and the run.

    ~/.local/share/ytpt/venv-asr/bin/python scripts/local_asr.py gr-sanleandro-3225
    ... --model mlx-community/whisper-tiny      # faster, worse
    ... --register        # write captions='asr_local' so classify.py picks it up
    ... --audio-only      # fetch + transcode, no ASR (measure the download)
    ... --chunk-min 45    # smaller pieces if Metal still refuses the buffer

Audio comes through the Granicus adapter: the HLS master where the player page
has one, otherwise the progressive MP4 the 2015-era player only exposes via
ASX.php (granicus.fetch_archive_mp4 -- Richmond, San Leandro). Politeness is
the adapter's: >= 5 s between requests, one ffmpeg at a time. The whole meeting
is pulled once (`-vn`, audio only on disk; the video bytes still cross the
wire on HLS).

Nothing under output/ is deleted; the wav is kept unless --drop-audio.
"""
import argparse, json, math, subprocess, sys, tempfile, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))
import state          # noqa: E402
import granicus       # noqa: E402
import segment as sg  # noqa: E402

OUT = ROOT / "output" / "t2" / "asr"
DEFAULT_MODEL = "mlx-community/whisper-large-v3-turbo"     # docs/MODELS.md 4
SAMPLE_RATE = 16000
# mlx-whisper loads the WHOLE file as one Metal buffer, so a long meeting dies
# on `[metal::malloc] Attempting to allocate N bytes which is greater than the
# maximum allowed buffer size` -- measured 2026-09-23 on gr-richmond-6042 (7.53
# h -> a 4.35 GB request). Richmond's meetings average 5.8 h, so the bulk run
# transcribes in chunks and shifts each chunk's cue times back onto the
# meeting's clock. 90 min is ~0.35 GB of samples, a wide margin, and the split
# is a stream copy (no re-encode, no quality cost). Cutting mid-word at a
# boundary does not matter to a locate track.
CHUNK_S = 5400.0
# --register moves the row to 'captioned' from these; any other status belongs
# to a stage that has already run (the cutter, the GPU), so the track is
# recorded without touching the status -- see the register branch in main().
REGISTER_FROM = ("no_episodes", "discovered", "classified")


def log(msg):
    print(f"{state.now_iso()} {msg}", flush=True)


def media_url(row, meta):
    """(url, kind, n_requests) for one Granicus meeting."""
    if meta.get("m3u8"):
        return meta["m3u8"], "m3u8", 0
    if meta.get("mp4"):
        return meta["mp4"], "mp4", 0
    site, view, clip = meta.get("site"), meta.get("view_id"), meta.get("clip_id")
    if not (site and view and clip):
        return None, "none", 0
    pj = ROOT / "output" / "subs" / f"{row['video_id']}.player.json"
    token = None
    if pj.exists():
        try:
            token = json.loads(pj.read_text(encoding="utf-8")).get("asx_token")
        except ValueError:
            pass
    url = granicus.fetch_archive_mp4(site, view, clip, token)
    return url, "asx-mp4", 1


def fetch_audio(url, dest, duration_s=None, timeout=4 * 3600):
    """Whole meeting -> 16 kHz mono wav (what whisper resamples to anyway).
    Returns (ok, detail, wall_s)."""
    cmd = ["ffmpeg", "-nostdin", "-v", "error", "-y", "-user_agent", granicus.UA, "-i", url,
           "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE), "-c:a", "pcm_s16le", str(dest)]
    granicus.polite_wait()
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    granicus._last_request["at"] = time.time()
    wall = time.time() - t0
    if r.returncode != 0 or not Path(dest).exists():
        return False, ((r.stderr or "").strip().splitlines() or ["ffmpeg failed"])[-1], wall
    return True, f"{Path(dest).stat().st_size / 1e6:.0f} MB wav", wall


def wav_duration(path):
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1", str(path)],
                       capture_output=True, text=True, timeout=120)
    return float(r.stdout.strip()) if r.returncode == 0 and r.stdout.strip() else None


def chunk_offsets(duration_s, chunk_s):
    """Where each piece starts. [] when the file is short enough to go whole."""
    if not duration_s or duration_s <= chunk_s or chunk_s <= 0:
        return []
    return [i * chunk_s for i in range(int(math.ceil(duration_s / chunk_s)))]


def shift_segments(segments, offset_s):
    """A chunk's cues moved back onto the meeting's clock."""
    out = []
    for s in segments:
        s = dict(s)
        s["start"] = float(s.get("start", 0.0)) + offset_s
        s["end"] = float(s.get("end", 0.0)) + offset_s
        out.append(s)
    return out


def split_wav(path, chunk_s, workdir):
    """[(offset_s, Path)] -- the wav as pieces of at most chunk_s, stream-copied
    into workdir. One piece (the file itself) when it is already short enough."""
    offsets = chunk_offsets(wav_duration(path), chunk_s)
    if not offsets:
        return [(0.0, Path(path))]
    parts = []
    for i, a in enumerate(offsets):
        dst = Path(workdir) / f"{Path(path).stem}.part{i:02d}.wav"
        r = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-ss", f"{a:.3f}", "-i", str(path),
                            "-t", f"{chunk_s:.3f}", "-c", "copy", str(dst)],
                           capture_output=True, text=True, timeout=1800)
        if r.returncode != 0 or not dst.exists():
            raise RuntimeError(f"chunking failed at {a:.0f} s: {(r.stderr or '').strip()[-200:]}")
        parts.append((a, dst))
    return parts


def transcribe(path, model, language="en", chunk_s=CHUNK_S):
    """mlx-whisper on Apple Silicon. Returns (segments, wall_s). No word
    timestamps and condition_on_previous_text off: cue-level times are all the
    segmenter needs, and the repetition loops that flag costs us are exactly
    what a locate pass must not have. Long meetings go through in chunks
    (CHUNK_S) with every cue shifted back onto the meeting's clock, so the VTT
    is indistinguishable from a single-pass one."""
    import mlx_whisper                                     # noqa: E402 -- only in the ASR venv
    t0 = time.time()
    segs = []
    with tempfile.TemporaryDirectory(prefix="ytpt-asr-") as wd:
        parts = split_wav(path, chunk_s, wd)
        for i, (offset, part) in enumerate(parts, 1):
            if len(parts) > 1:
                log(f"  chunk {i}/{len(parts)} from {offset / 60:.0f} min")
            res = mlx_whisper.transcribe(str(part), path_or_hf_repo=model, language=language,
                                         condition_on_previous_text=False, word_timestamps=False, verbose=None)
            segs += shift_segments(res.get("segments") or [], offset)
    segs.sort(key=lambda s: s["start"])
    return segs, time.time() - t0


def detected_language(segments, fallback="en"):
    """With --language auto, name the track from its own text, using the same
    detector caption_audit and stage 1 use (segment.guess_language)."""
    text = " ".join(str(s.get("text", "")) for s in segments[:400])
    try:
        g = sg.guess_language(text)
    except Exception:
        return fallback
    return g if g in ("en", "es") else fallback


def vtt_time(t):
    t = max(0.0, float(t))
    h, rem = divmod(t, 3600)
    m, s = divmod(rem, 60)
    return f"{int(h):02d}:{int(m):02d}:{s:06.3f}"


def write_vtt(segments, dest, language="en"):
    """A caption-equivalent track: segment.vtt_to_segments reads this unchanged.
    The Language header has to say what was actually transcribed -- classify.py
    picks its keyword files off the track's own language."""
    lines = ["WEBVTT", "Kind: captions", f"Language: {language}", ""]
    n = 0
    for s in segments:
        text = " ".join(str(s.get("text", "")).split())
        if not text:
            continue
        lines += [f"{vtt_time(s['start'])} --> {vtt_time(s['end'])}", text, ""]
        n += 1
    Path(dest).write_text("\n".join(lines), encoding="utf-8")
    return n


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video_id")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--register", action="store_true", help="write captions='asr_local' + path and move the row to 'captioned'")
    ap.add_argument("--audio-only", action="store_true", help="fetch and transcode only, no ASR")
    ap.add_argument("--drop-audio", action="store_true", help="remove the working wav when done (it is large)")
    ap.add_argument("--chunk-min", type=float, default=CHUNK_S / 60,
                    help="transcribe in pieces of this many minutes (mlx-whisper's Metal buffer limit)")
    ap.add_argument("--language", default="en",
                    help="ASR language, and the VTT's Language header (default en; 'auto' lets whisper detect)")
    ap.add_argument("--state", default=None)
    a = ap.parse_args(argv)

    con = state.connect(a.state)
    row = con.execute("SELECT * FROM videos WHERE video_id = ?", (a.video_id,)).fetchone()
    if not row:
        sys.exit(f"no video {a.video_id}")
    meta = json.loads(row["raw_meta"] or "{}")
    OUT.mkdir(parents=True, exist_ok=True)
    wav = OUT / f"{a.video_id}.wav"
    lang = None if a.language == "auto" else a.language
    vtt = OUT / f"{a.video_id}.{a.language}.vtt"     # renamed below once 'auto' resolves
    report = {"video_id": a.video_id, "body": row["speaker"], "date": row["meeting_date"],
              "catalog_duration_s": row["duration_s"], "model": a.model, "at": state.now_iso()}

    if wav.exists():
        log(f"audio already on disk: {wav} ({wav.stat().st_size / 1e6:.0f} MB)")
        report["fetch_s"] = 0.0
    else:
        url, kind, nreq = media_url(row, meta)
        if not url:
            sys.exit(f"{a.video_id}: no m3u8 / mp4 / ASX media URL; cannot fetch audio")
        log(f"audio via {kind}: {url[:110]}")
        ok, detail, wall = fetch_audio(url, wav, row["duration_s"])
        if not ok:
            sys.exit(f"{a.video_id}: audio fetch failed after {wall:.0f} s: {detail}")
        log(f"fetched {detail} in {wall / 60:.1f} min ({kind}, {nreq + 1} request(s))")
        report.update(fetch_s=round(wall, 1), media_kind=kind, media_url=url)

    dur = wav_duration(wav) or (row["duration_s"] or 0)
    report["audio_s"] = round(dur, 1)
    report["wav_mb"] = round(wav.stat().st_size / 1e6, 1)
    log(f"audio: {dur / 3600:.2f} h, {report['wav_mb']:.0f} MB, {SAMPLE_RATE} Hz mono")
    if a.audio_only:
        print(json.dumps(report, indent=1))
        return 0

    log(f"transcribing with {a.model} ({a.language}) ...")
    segs, wall = transcribe(wav, a.model, language=lang, chunk_s=a.chunk_min * 60)
    track_lang = lang or detected_language(segs)
    vtt = OUT / f"{a.video_id}.{track_lang}.vtt"
    n_cues = write_vtt(segs, vtt, language=track_lang)
    report["language"] = track_lang
    rtf = dur / wall if wall else 0
    report.update(asr_s=round(wall, 1), segments=len(segs), cues=n_cues, realtime_factor=round(rtf, 2),
                  vtt=str(vtt), words=sum(len(str(s.get("text", "")).split()) for s in segs))
    (OUT / f"{a.video_id}.asr.json").write_text(json.dumps({"model": a.model, "segments": segs}, ensure_ascii=False), encoding="utf-8")
    log(f"ASR done: {len(segs)} segments, {report['words']} words, {wall / 60:.1f} min wall for "
        f"{dur / 3600:.2f} h of audio = {rtf:.1f}x realtime -> {vtt}")

    if a.register:
        note = f"local ASR locate track: {a.model}, {n_cues} cues, {rtf:.1f}x realtime"
        fields = {"captions": "asr_local", "captions_path": str(vtt),
                  "raw_meta": json.dumps({**meta, "asr": {"model": a.model, "realtime_factor": report["realtime_factor"],
                                                          "cues": n_cues, "at": report["at"]}}, ensure_ascii=False)}
        if row["status"] in REGISTER_FROM:
            state.transition(con, a.video_id, row["status"], "captioned", note=note, **fields)
            log(f"registered: {a.video_id} is now 'captioned' with captions='asr_local' (classify.py will read {vtt.name})")
        else:
            # A row another stage has advanced ('cut', 'downloaded', 'transcribing'
            # -- Hayward has agenda_index episodes already cut) keeps its status:
            # only `classify.py --include-classified` knows whether this track
            # locates anything new, and finish_status owns the move back. The
            # track is STILL recorded, or the row would look untranscribed and
            # scripts/asr_batch.py would fetch the same meeting again forever.
            con.execute("UPDATE videos SET captions = ?, captions_path = ?, raw_meta = ?, updated_at = ? WHERE video_id = ?",
                        (fields["captions"], fields["captions_path"], fields["raw_meta"], state.now_iso(), a.video_id))
            con.execute("INSERT INTO events (video_id, from_status, to_status, at, note) VALUES (?,?,?,?,?)",
                        (a.video_id, row["status"], row["status"], state.now_iso(), f"local_asr: {note}"[:300]))
            con.commit()
            log(f"recorded on an advanced row: {a.video_id} stays '{row['status']}' with captions='asr_local'; "
                f"`classify.py run --include-classified` reads {vtt.name}")
    (OUT / f"{a.video_id}.report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    if a.drop_audio:
        wav.unlink()
        log(f"removed the working wav {wav.name}")
    print(json.dumps(report, indent=1))
    con.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
