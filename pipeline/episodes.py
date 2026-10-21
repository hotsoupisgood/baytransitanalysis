#!/usr/bin/env python3
"""
pipeline/episodes.py -- meeting stage 3: 'classified' -> 'downloading' ->
'downloaded' -> 'cut' (PROPOSAL-transit.md 3e revised by T0 test 4; built
2026-09-21). Turns every 'located' episode of a meeting into a clip that the
GPU batch machinery already consumes.

  youtube   the WHOLE meeting audio, once (`yt-dlp -f bestaudio -x
            --audio-format opus`, one player request per meeting: T0 test 4
            run 2, 469x real time, zero 403s -- the per-episode
            --download-sections pattern is what got a 403 after 5 opens) into
            output/audio/<id>.opus, then every episode cut locally with
            `ffmpeg -ss -t -c copy` (packet-accurate on Ogg Opus, ~20 ms). A
            meeting whose whole audio T0 already fetched
            (output/t0/<id>/full.opus) is symlinked, not re-requested.
  granicus  the HLS rendition is video+audio (410 MB per meeting-hour on the
            wire, T0 test 7) and `-ss` seeks on it, so each episode is an
            audio-only SLICE (`ffmpeg -ss -t -vn -c:a copy` on the m3u8,
            pipeline/granicus.audio_slice); no whole meeting, no MP4.
  local wav a Granicus meeting with NO m3u8 (Richmond, San Leandro: media only
            via ASX.php -> MP4, plan/T2-PLAN.md 3d) whose whole audio the local
            ASR run already pulled to output/t2/asr/<id>.wav (16 kHz mono pcm,
            scripts/local_asr.py) is cut from THAT file, offline, exactly like
            the YouTube path (copy_cut -> .wav clip -> register_clip). Found
            2026-10-16: 94 'located' local_asr episodes sat behind a no-op
            because cut_granicus had nothing to slice. The cut source is
            recorded on the clip row as raw_meta.cut_source = 'local_wav';
            episodes.method stays the LOCATE route and is never touched here.

  bounds    core window +- PAD_S (30 s), clamped to the meeting, then each
            edge SNAPPED to the nearest caption-cue edge within SNAP_MAX_S
            (T0 edge check: the nearest cue edge is median 0.6 s, p90 1.3 s
            away; the three boundaries > 5 s from any cue were inside a
            recess and stay where they are).
  register  clip -> 16 kHz mono 16 kbps opus with transcode.ffmpeg_command
            (cap = the clip's own length) into output/opus/<id>_e<n>.opus, and
            ONE `videos` row per clip, kind='episode', speaker = body, walked
            discovered -> downloaded -> transcoded with state.transition and
            flipped in_frame = 1 only then (exactly scripts/t0_register_clips.py:
            the launchd workers select in-frame rows, so they never see a
            half-registered one). gpu_batch.propose --speaker <bodies> takes
            it from there. A clip row that already exists (the T0 test-5
            clips) is linked, never re-cut.

  retention the whole-meeting audio is KEPT (default; 45 MB per meeting-hour
            = 138 GB for the 12-month window, T0 test 8). `--drop-full` is
            deliberately absent: nothing under output/ is deleted by code.

YouTube rules (AGENTS.md): this Mac only, sequential, --sleep-requests 2,
random 45-180 s gaps between meetings, yt-dlp retries capped at 2; STOP (exit
3, row back to 'classified') on "Sign in to confirm", 429 or a second 403.
Granicus: >= 5 s between requests (granicus.GAP_S), one ffmpeg at a time; a
403/429 from the stream host stops the run the same way (episode stays
'located', meeting back to 'classified', exit 3).

    python3 pipeline/episodes.py --plan                # what would be fetched / cut, no network
    python3 pipeline/episodes.py --max 3               # this many meetings, then exit
    python3 pipeline/episodes.py --bodies oakland
"""
import argparse, json, os, random, re, subprocess, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import state, transcode, discover, granicus, segment as sg  # noqa: E402
from downloader import ytdlp_bin, BOT_CHECK, FORBIDDEN, error_line, random_gap, audio_dir, ffprobe_duration  # noqa: E402

ROOT = state.ROOT
T0 = ROOT / "output" / "t0"
PAD_S = 30.0
CUT_LOCAL_WAV = "local_wav"      # raw_meta.cut_source on a clip cut from the local ASR wav
SNAP_MAX_S = 3.0
COPY_TOL_S = 1.0             # copy-cut duration error above this -> re-encode the cut
EXIT_BOT_CHECK = 3
NOTE = "episode clip"
# ffmpeg's wording when the HLS host refuses: "Server returned 403 Forbidden (access denied)" / 429
HLS_BLOCKED = re.compile(r"Server returned (403|429)|HTTP error (403|429)|Too Many Requests", re.I)
# Which meetings have cutting left to do. A meeting the cutter already finished
# can gain new 'located' episodes when a SECOND instrument reads it (classify.py
# --include-classified on a row agenda_index had moved to 'classified', or a
# local ASR track on a meeting cut from its captions: 2026-09-23), so a 'cut'
# row with work left is selected too and claimed back to 'downloaded'.
SELECT_WHERE = ("kind = 'meeting' AND (status IN ('classified', 'downloaded') OR (status = 'cut' AND EXISTS "
                "(SELECT 1 FROM episodes e WHERE e.video_id = videos.video_id AND e.status = 'located')))")


def hms(s):
    s = int(round(s))
    return f"{s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d}"


# --- bounds ----------------------------------------------------------------------

def cue_edges(segments):
    return sorted({float(s["start"]) for s in segments} | {float(s["end"]) for s in segments})


def snap(t, edges, max_dist=SNAP_MAX_S):
    """The nearest cue edge to t if it is within max_dist, else t."""
    if not edges:
        return t
    import bisect
    i = bisect.bisect_left(edges, t)
    best = min((e for e in edges[max(0, i - 1):i + 2]), key=lambda e: abs(e - t), default=None)
    return best if best is not None and abs(best - t) <= max_dist else t


def clip_bounds(core_start, core_end, duration_s, edges, pad=PAD_S):
    """Padded, clamped, cue-snapped [start, end] for one episode."""
    a = max(0.0, core_start - pad)
    b = min(float(duration_s), core_end + pad) if duration_s else core_end + pad
    a, b = snap(a, edges), snap(b, edges)
    if b - a < 1.0:
        b = a + 1.0
    return round(a, 3), round(b, 3)


# --- fetch -----------------------------------------------------------------------

def ytdlp_whole_command(video_id, out_dir):
    """T0 test 4 run 2 (scripts/t0_fetch_episodes.py fetch_whole), verbatim
    except for the output dir: bestaudio, opus container (format 251 is
    copied, not re-encoded), yt-dlp's own retries capped at 2."""
    return [ytdlp_bin(), "--sleep-requests", "2", "--no-playlist", "--no-progress", "--retries", "2", "--fragment-retries", "2",
            "-f", "bestaudio", "-x", "--audio-format", "opus",
            "-o", str(Path(out_dir) / "%(id)s.%(ext)s"), f"https://www.youtube.com/watch?v={video_id}"]


def fetch_whole_youtube(video_id, out_dir, consecutive_403=0, timeout=6 * 3600):
    """('ok'|'error'|'bot', path or message, made_request)."""
    dst = Path(out_dir) / f"{video_id}.opus"
    if dst.exists():
        return "ok", dst, False
    t0_full = T0 / video_id / "full.opus"
    if t0_full.exists():
        dst.symlink_to(os.path.relpath(t0_full, dst.parent))
        return "ok", dst, False
    t0 = time.time()
    try:
        r = subprocess.run(ytdlp_whole_command(video_id, out_dir), capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return "error", f"yt-dlp timed out after {timeout} s", True
    if r.returncode != 0:
        err = (r.stderr or "") + (r.stdout or "")
        if BOT_CHECK.search(err) or (FORBIDDEN.search(err) and consecutive_403 + 1 >= 2):
            return "bot", error_line(err), True
        return "error", error_line(err), True
    if not dst.exists():
        cands = [p for p in Path(out_dir).glob(f"{video_id}.*") if p.suffix not in (".json", ".part", ".ytdl")]
        if not cands:
            return "error", "yt-dlp exited 0 but wrote no audio file", True
        dst = cands[0]
    return "ok", dst, True


# --- cut ---------------------------------------------------------------------------

def copy_cut(src, a, b, dst):
    """ffmpeg -ss/-t -c copy; re-encode (libopus 112k) if the copy is off by
    more than COPY_TOL_S (T0: every Ogg copy-cut landed within 20 ms)."""
    for codec in (["-c", "copy"], ["-c:a", "libopus", "-b:a", "112k"]):
        r = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-ss", f"{a:.3f}", "-i", str(src), "-t", f"{b - a:.3f}",
                            *codec, str(dst)], capture_output=True, text=True, timeout=1800)
        d = ffprobe_duration(dst) if Path(dst).exists() else None
        if r.returncode == 0 and d is not None and abs(d - (b - a)) <= COPY_TOL_S:
            return True, ("copy" if codec[0] == "-c" else "reencode"), d
    return False, (r.stderr or "").strip()[-200:] or f"ffmpeg exit {r.returncode}", None


def register_clip(con, meeting, ep, clip_path, a, b, opus_dir, note=NOTE, cut_source=None):
    """The T0 test-5 registration, generalised: opus + one in-frame 'transcoded'
    videos row (kind='episode') for one cut episode. Returns (status_text, clip_video_id).
    `cut_source` (2026-10-16) names where the clip's audio came from when that
    is not the platform's own route (CUT_LOCAL_WAV); it lands in raw_meta only."""
    clip_id = ep["episode_id"]
    row = con.execute("SELECT status, in_frame, raw_meta FROM videos WHERE video_id = ?", (clip_id,)).fetchone()
    if row:
        con.execute("UPDATE videos SET kind = 'episode', episode_id = ?, meeting_id = ? WHERE video_id = ?",
                    (clip_id, meeting["meeting_id"], clip_id))
        return f"exists ({row['status']}, in_frame={row['in_frame']}); linked", clip_id
    dst = Path(opus_dir) / f"{clip_id}.opus"
    if dst.exists():
        return f"REFUSED: {dst} exists but no videos row {clip_id} -- collision", None
    dur = ffprobe_duration(clip_path)
    if not dur:
        return f"REFUSED: ffprobe cannot read {clip_path}", None
    tmp = Path(opus_dir) / f"{clip_id}.tmp.opus"
    # transcode.encode measures the mono signal first and inserts a single-channel
    # downmix and/or loudnorm when the measurement asks for it (transcode.py
    # "LEVEL"); `levels` is kept on the row so a QA failure can be read back.
    ok, enc_detail, levels = transcode.encode(clip_path, tmp, dur)
    if not ok:
        tmp.unlink(missing_ok=True)
        return f"REFUSED: ffmpeg transcode failed: {enc_detail}", None
    ok, detail = transcode.verify(tmp, dur)
    if not ok:
        tmp.unlink(missing_ok=True)
        return f"REFUSED: opus verify failed: {detail}", None
    tmp.replace(dst)
    detail = f"{detail}, {enc_detail}"
    # '<meeting>_e<n>' (classifier) or '<meeting>_a<n>' (agenda_index, 2026-09-22)
    m_n = re.search(r"_([a-z])(\d+)$", clip_id)
    n = int(m_n.group(2)) if m_n else 0
    raw = {"episode_id": clip_id, "source_video_id": meeting["video_id"], "meeting_id": meeting["meeting_id"],
           "label": ep["label"], "method": ep.get("method"), "tags": ep["tags"], "window_start_s": a, "window_end_s": b,
           "core_start_s": ep["core_start_s"], "core_end_s": ep["core_end_s"], "agenda_item": ep.get("agenda_item"),
           "platform": meeting["platform"], "meeting_duration_s": meeting["duration_s"], "meeting_title": meeting["title"],
           "prompt_version": ep.get("prompt_version"), "classifier_model": ep.get("classifier_model"),
           "audio_levels": levels}
    if cut_source:
        raw["cut_source"] = cut_source
    now = state.now_iso()
    con.execute("BEGIN IMMEDIATE")
    con.execute(
        "INSERT INTO videos (video_id, speaker, channel_url, month, title, upload_date, date_exact, duration_s, raw_meta, "
        "crawled_at, in_frame, status, attempts, updated_at, sponsor_segments, kind, platform, source_url, meeting_id, meeting_date, episode_id) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,0,'discovered',0,?,'[]','episode',?,?,?,?,?)",
        (clip_id, meeting["speaker"], meeting["channel_url"], meeting["month"],
         f"{meeting['title']} — episode {n} ({ep['label'] or ep.get('method') or 'located'}) {hms(a)}–{hms(b)}", meeting["upload_date"],
         meeting["date_exact"] or 0, round(dur, 2), json.dumps(raw, ensure_ascii=False), now, now,
         meeting["platform"], meeting["source_url"], meeting["meeting_id"], meeting["meeting_date"], clip_id))
    con.execute("INSERT INTO events (video_id, from_status, to_status, at, note) VALUES (?,?,?,?,?)",
                (clip_id, None, "discovered", now, note))
    con.execute("COMMIT")
    if not state.transition(con, clip_id, "discovered", "downloaded", note=note, audio_path=str(clip_path)):
        return f"REFUSED: {clip_id} could not move discovered -> downloaded", None
    if not state.transition(con, clip_id, "downloaded", "transcoded", note=f"{note}: {detail}", opus_path=str(dst), transcribed_s=dur):
        return f"REFUSED: {clip_id} could not move downloaded -> transcoded", None
    discover.mark_frame(con, [clip_id], note)
    return f"transcoded {detail}", clip_id


def episode_status_from_clip(con, ep_id):
    """After linking an existing clip row: the episodes row follows the clip."""
    row = con.execute("SELECT status, audio_path, opus_path, raw_meta FROM videos WHERE video_id = ?", (ep_id,)).fetchone()
    if not row:
        return
    m = {"transcoded": "cut", "queued": "queued", "transcribing": "transcribing", "done": "done", "failed": "failed", "dead": "failed"}
    st = m.get(row["status"], "cut")
    fields = {"audio_path": row["audio_path"], "opus_path": row["opus_path"]}
    try:
        meta = json.loads(row["raw_meta"] or "{}")
        if meta.get("window_start_s") is not None:
            fields["start_s"], fields["end_s"] = float(meta["window_start_s"]), float(meta["window_end_s"])
    except (ValueError, TypeError):
        pass
    if st == "done":
        tp = ROOT / "output" / "whisper" / f"{ep_id}.json"
        fields["transcript_path"] = str(tp) if tp.exists() else None
    state.set_episode(con, ep_id, st, note="linked to existing clip row", **fields)


def located_episodes(con, video_id):
    return [dict(r) for r in con.execute("SELECT * FROM episodes WHERE video_id = ? AND status = 'located' ORDER BY start_s", (video_id,))]


def caption_segments(meeting):
    p = meeting["captions_path"]
    return sg.read_segments(p) if p and Path(p).exists() else []


def cut_youtube(con, meeting, full_path, out_dir, opus_dir, log):
    """Every located episode of one meeting from its whole audio. Returns (n_ok, n_fail, seconds)."""
    edges = cue_edges(caption_segments(meeting))
    dur_full = ffprobe_duration(full_path) or meeting["duration_s"]
    ok = bad = 0
    secs = 0.0
    for ep in located_episodes(con, meeting["video_id"]):
        exists = con.execute("SELECT 1 FROM videos WHERE video_id = ?", (ep["episode_id"],)).fetchone()
        if exists:
            msg, _ = register_clip(con, meeting, ep, None, ep["start_s"], ep["end_s"], opus_dir)
            episode_status_from_clip(con, ep["episode_id"])
            log(f"  {ep['episode_id']}: {msg}")
            ok += 1
            continue
        a, b = clip_bounds(ep["core_start_s"], ep["core_end_s"], dur_full, edges)
        clip = Path(out_dir) / f"{ep['episode_id']}.opus"
        if not clip.exists():
            good, how, d = copy_cut(full_path, a, b, clip)
            if not good:
                state.set_episode(con, ep["episode_id"], "failed", note=f"cut: {how}", last_error=str(how)[:300])
                log(f"  {ep['episode_id']}: cut FAILED {how}")
                bad += 1
                continue
        else:
            how, d = "on disk", ffprobe_duration(clip)
        msg, clip_id = register_clip(con, meeting, ep, clip, a, b, opus_dir)
        if not clip_id:
            state.set_episode(con, ep["episode_id"], "failed", note=msg[:200], last_error=msg[:300])
            log(f"  {ep['episode_id']}: {msg}")
            bad += 1
            continue
        state.set_episode(con, ep["episode_id"], "cut", note=f"{how}, {d:.1f} s", start_s=a, end_s=b,
                          audio_path=str(clip), opus_path=str(Path(opus_dir) / f"{clip_id}.opus"))
        secs += d or 0
        ok += 1
        log(f"  {ep['episode_id']} {hms(a)}-{hms(b)} ({(b - a) / 60:.1f} min, {how}): {msg}")
    return ok, bad, secs


def cut_granicus(con, meeting, out_dir, opus_dir, log, dry_run=False):
    """Per-episode audio slices from the HLS master; returns (n_ok, n_fail,
    seconds, MB, n_requests, blocked). `blocked` is the ffmpeg error line when
    the stream host answered 403/429: the loop stops there, the episode stays
    'located' (not charged), and run() treats it like a YouTube bot check."""
    meta = json.loads(meeting["raw_meta"] or "{}")
    m3u8 = meta.get("m3u8")
    if not m3u8:
        return 0, 0, 0.0, 0.0, 0, None
    edges = cue_edges(caption_segments(meeting))
    ok = bad = nreq = 0
    secs = mb = 0.0
    for ep in located_episodes(con, meeting["video_id"]):
        if con.execute("SELECT 1 FROM videos WHERE video_id = ?", (ep["episode_id"],)).fetchone():
            msg, _ = register_clip(con, meeting, ep, None, ep["start_s"], ep["end_s"], opus_dir)
            episode_status_from_clip(con, ep["episode_id"])
            log(f"  {ep['episode_id']}: {msg}")
            ok += 1
            continue
        a, b = clip_bounds(ep["core_start_s"], ep["core_end_s"], meeting["duration_s"], edges)
        clip = Path(out_dir) / f"{ep['episode_id']}.m4a"
        if dry_run:
            log(f"  DRY RUN {ep['episode_id']} {hms(a)}-{hms(b)} (timeout {granicus.slice_timeout(b - a):.0f} s): "
                + " ".join(granicus.audio_slice_command(m3u8, a, b - a, granicus.part_path(clip))))
            continue
        if not clip.exists():
            good, detail, wall = granicus.audio_slice(m3u8, a, b - a, clip)
            nreq += 1
            if not good and HLS_BLOCKED.search(" ".join(detail) if isinstance(detail, list) else str(detail)):
                log(f"  {ep['episode_id']}: stream host refused the slice ({detail}); stopping, episode stays 'located'")
                return ok, bad, secs, mb, nreq, str(detail)[:200]
            if not good:
                state.set_episode(con, ep["episode_id"], "failed", note=f"slice: {detail}", last_error=str(detail)[:300])
                log(f"  {ep['episode_id']}: slice FAILED {detail}")
                bad += 1
                continue
            time.sleep(random.uniform(5, 15))
        d = ffprobe_duration(clip) or 0.0
        # -ss on HLS lands on a segment boundary before the request: the slice
        # ends on time and starts early by up to one segment (T0: 2-s segments)
        a_actual = round(b - d, 3) if d > (b - a) + 0.5 else a
        msg, clip_id = register_clip(con, meeting, ep, clip, a_actual, b, opus_dir)
        if not clip_id:
            state.set_episode(con, ep["episode_id"], "failed", note=msg[:200], last_error=msg[:300])
            log(f"  {ep['episode_id']}: {msg}")
            bad += 1
            continue
        state.set_episode(con, ep["episode_id"], "cut", note=f"hls slice, {d:.1f} s, {clip.stat().st_size / 1e6:.1f} MB",
                          start_s=a_actual, end_s=b, audio_path=str(clip), opus_path=str(Path(opus_dir) / f"{clip_id}.opus"))
        secs += d; mb += clip.stat().st_size / 1e6
        ok += 1
        log(f"  {ep['episode_id']} {hms(a_actual)}-{hms(b)} ({d / 60:.1f} min, {clip.stat().st_size / 1e6:.1f} MB): {msg}")
    return ok, bad, secs, mb, nreq, None


# --- local wav fallback (2026-10-16) -------------------------------------------------

def local_asr_dir():
    """Where scripts/local_asr.py leaves the whole-meeting wav (LOCAL_ASR_DIR
    overrides it, for tests)."""
    return Path(os.path.expanduser(state.setting("LOCAL_ASR_DIR", str(ROOT / "output" / "t2" / "asr"))))


def local_wav_path(meeting):
    """The whole-meeting wav the local ASR run pulled for a Granicus meeting
    that has NO m3u8, or None. A meeting WITH an m3u8 keeps the HLS slice
    route even if a wav happens to exist, so nothing already working changes."""
    if meeting.get("platform") != "granicus":
        return None
    try:
        meta = json.loads(meeting.get("raw_meta") or "{}")
    except (ValueError, TypeError):
        meta = {}
    if meta.get("m3u8"):
        return None
    p = local_asr_dir() / f"{meeting['video_id']}.wav"
    return p if p.exists() else None


def wav_label(p):
    """'output/t2/asr/<id>.wav' when under the repo, else the absolute path."""
    try:
        return str(Path(p).resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(p)


def cut_local_wav(con, meeting, wav_path, out_dir, opus_dir, log):
    """Every located episode of one Granicus meeting from the local ASR wav
    (offline; no request). Same bounds as the other two routes: core +- PAD_S,
    clamped to the wav's own length, snapped to the meeting's caption cues
    (here the ASR track's). The clip is a `-c copy` .wav slice (sample-exact
    on pcm) that register_clip transcodes to the 16 kHz mono opus the GPU
    consumes. Returns (n_ok, n_fail, seconds)."""
    edges = cue_edges(caption_segments(meeting))
    dur_full = ffprobe_duration(wav_path) or meeting["duration_s"]
    ok = bad = 0
    secs = 0.0
    for ep in located_episodes(con, meeting["video_id"]):
        if con.execute("SELECT 1 FROM videos WHERE video_id = ?", (ep["episode_id"],)).fetchone():
            msg, _ = register_clip(con, meeting, ep, None, ep["start_s"], ep["end_s"], opus_dir)
            episode_status_from_clip(con, ep["episode_id"])
            log(f"  {ep['episode_id']}: {msg}")
            ok += 1
            continue
        a, b = clip_bounds(ep["core_start_s"], ep["core_end_s"], dur_full, edges)
        clip = Path(out_dir) / f"{ep['episode_id']}.wav"
        if not clip.exists():
            good, how, d = copy_cut(wav_path, a, b, clip)
            if not good:
                state.set_episode(con, ep["episode_id"], "failed", note=f"cut (local wav): {how}", last_error=str(how)[:300])
                log(f"  {ep['episode_id']}: cut FAILED {how}")
                bad += 1
                continue
        else:
            how, d = "on disk", ffprobe_duration(clip)
        msg, clip_id = register_clip(con, meeting, ep, clip, a, b, opus_dir, cut_source=CUT_LOCAL_WAV)
        if not clip_id:
            state.set_episode(con, ep["episode_id"], "failed", note=msg[:200], last_error=msg[:300])
            log(f"  {ep['episode_id']}: {msg}")
            bad += 1
            continue
        state.set_episode(con, ep["episode_id"], "cut", note=f"local wav, {how}, {d:.1f} s", start_s=a, end_s=b,
                          audio_path=str(clip), opus_path=str(Path(opus_dir) / f"{clip_id}.opus"))
        secs += d or 0
        ok += 1
        log(f"  {ep['episode_id']} {hms(a)}-{hms(b)} ({(b - a) / 60:.1f} min, local wav {how}): {msg}")
    return ok, bad, secs


def log(msg):
    print(f"{state.now_iso()} {msg}", flush=True)


def requeue_failed(con, bodies=None):
    """'failed' episodes (cut / slice errors) back to 'located', and their
    meeting back to 'downloaded' so the next run cuts them again. A human's
    explicit request, like pipeline/requeue.py; never automatic."""
    where = "e.status = 'failed'"
    params = []
    if bodies:
        where += f" AND v.speaker IN ({','.join('?' * len(bodies))})"
        params = bodies
    rows = con.execute(f"SELECT e.episode_id, e.video_id, v.status AS vstatus FROM episodes e JOIN videos v USING (video_id) WHERE {where}", params).fetchall()
    n = 0
    for r in rows:
        if state.set_episode(con, r["episode_id"], "located", note="requeued by human", last_error=None):
            n += 1
        if r["vstatus"] == "cut":
            state.transition(con, r["video_id"], "cut", "downloaded", note="episode requeued")
    return n


def run(args):
    con = state.connect(args.state)
    if args.requeue_failed:
        n = requeue_failed(con, [b.strip() for b in args.bodies.split(",")] if args.bodies else None)
        log(f"requeued {n} failed episode(s) to 'located'")
        con.close()
        return 0
    out_dir, opus_dir = audio_dir(), transcode.opus_dir()
    out_dir.mkdir(parents=True, exist_ok=True); opus_dir.mkdir(parents=True, exist_ok=True)
    bodies = [b.strip() for b in args.bodies.split(",") if b.strip()] if args.bodies else []
    # kind='meeting' only: a show row the launchd downloader is fetching right
    # now is also 'downloading' and must not be pushed into a meeting state.
    # --bodies only: another episodes.py run on OTHER bodies may be mid-fetch.
    n = state.reclaim_stuck(con, "downloading", "classified", note="reclaimed at episodes.py start", kind="meeting",
                            speakers=bodies or None)
    if n:
        log(f"reclaimed {n} meeting row(s) left in 'downloading'")
    where = SELECT_WHERE
    params = []
    if bodies:
        where += f" AND speaker IN ({','.join('?' * len(bodies))})"
        params = bodies
    rows = [dict(r) for r in con.execute(f"SELECT * FROM videos WHERE {where} ORDER BY meeting_date DESC, video_id"
                                         + (f" LIMIT {int(args.max)}" if args.max else ""), params)]
    log(f"{len(rows)} meeting(s) with located episodes -> audio {out_dir}, clips {opus_dir}" + (" (plan only)" if args.plan else ""))
    rng = random.Random()
    stats = {"meetings": 0, "clips": 0, "failed": 0, "yt_requests": 0, "yt_bytes": 0, "yt_wall": 0.0, "gr_requests": 0, "gr_mb": 0.0,
             "clip_s": 0.0, "bot": None}
    consecutive_403 = 0
    for m in rows:
        vid = m["video_id"]
        eps = located_episodes(con, vid)
        if args.plan:
            wav = local_wav_path(m)
            src = ("T0 full.opus" if (T0 / vid / "full.opus").exists() else "on disk" if (out_dir / f"{vid}.opus").exists()
                   else f"local wav ({wav_label(wav)})" if wav else m["platform"])
            log(f"  {vid} ({m['speaker']}, {m['meeting_date']}): {len(eps)} episodes, {sum(e['end_s'] - e['start_s'] for e in eps) / 60:.0f} core min; audio via {src}")
            for e in eps:
                log(f"      {e['episode_id']} {hms(e['core_start_s'])}-{hms(e['core_end_s'])} {e['label']} {e['tags']}"
                    + (f" [{e['agenda_item'][:60]}]" if e.get("agenda_item") else ""))
            continue
        if m["status"] == "cut":
            # a second instrument located more episodes in a meeting already cut
            if not state.transition(con, vid, "cut", "downloaded",
                                    note=f"{len(eps)} new located episode(s) from a second instrument"):
                continue
            m["status"] = "downloaded"
        if m["platform"] == "youtube":
            if m["status"] == "classified":
                if not state.claim(con, vid, "classified", "downloading", count_attempt=True):
                    continue
                need = not (out_dir / f"{vid}.opus").exists() and not (T0 / vid / "full.opus").exists()
                if need and stats["yt_requests"]:
                    gap = random_gap(args.min_gap, args.max_gap, rng)
                    log(f"  sleeping {gap:.0f} s before the next YouTube request")
                    time.sleep(gap)
                t0 = time.time()
                kind, res, made = fetch_whole_youtube(vid, out_dir, consecutive_403)
                if made:
                    stats["yt_requests"] += 1; stats["yt_wall"] += time.time() - t0
                if kind == "bot":
                    stats["bot"] = res
                    state.transition(con, vid, "downloading", "classified", note="bot check / rate limit; not charged",
                                     attempts=max(0, int(m["attempts"] or 0)), last_error=res)
                    log(f"STOP: YouTube is challenging this IP on {vid}: {res}")
                    break
                if kind == "error":
                    consecutive_403 += 1 if FORBIDDEN.search(str(res)) else 0
                    new = state.fail(con, vid, "downloading", res)
                    log(f"{new.upper()} {vid}: {res}")
                    stats["failed"] += 1
                    continue
                full = Path(res)
                d = ffprobe_duration(full)
                if made:
                    stats["yt_bytes"] += full.stat().st_size
                state.transition(con, vid, "downloading", "downloaded", audio_path=str(full),
                                 note=f"whole audio {full.stat().st_size / 1e6:.0f} MB, {d or 0:.0f} s, {time.time() - t0:.0f} s wall"
                                      + ("" if made else " (T0 cache / on disk)"))
                log(f"downloaded {vid} ({m['speaker']}): {full.name} {full.stat().st_size / 1e6:.0f} MB, {(d or 0) / 3600:.2f} h, "
                    f"{time.time() - t0:.0f} s" + ("" if made else " -- reused, no request"))
                m["audio_path"] = str(full)
            full = Path(m["audio_path"] or (out_dir / f"{vid}.opus"))
            ok, bad, secs = cut_youtube(con, m, full, out_dir, opus_dir, log)
        else:
            if m["status"] == "classified":
                if not state.claim(con, vid, "classified", "downloading", count_attempt=True):
                    continue
            wav = local_wav_path(m)
            if wav:
                # no m3u8, but the local ASR run left the whole meeting on disk: cut offline, no request
                ok, bad, secs = cut_local_wav(con, m, wav, out_dir, opus_dir, log)
                mb, nreq, blocked = 0.0, 0, None
                fetched = f"local wav {wav_label(wav)}, no request"
            else:
                ok, bad, secs, mb, nreq, blocked = cut_granicus(con, m, out_dir, opus_dir, log)
                fetched = f"granicus: {nreq} audio slices, {mb:.1f} MB"
            stats["gr_requests"] += nreq; stats["gr_mb"] += mb
            if blocked:
                # same rule as the YouTube path: stop, do not charge, a human decides
                stats["bot"] = blocked
                if m["status"] == "classified":
                    state.transition(con, vid, "downloading", "classified", note="granicus 403/429; not charged",
                                     attempts=max(0, int(m["attempts"] or 0)), last_error=blocked)
                log(f"STOP: the Granicus stream host is refusing slices on {vid}: {blocked}")
                break
            if m["status"] == "classified":
                state.transition(con, vid, "downloading", "downloaded", note=fetched)
        stats["clips"] += ok; stats["failed"] += bad; stats["clip_s"] += secs; stats["meetings"] += 1
        left = con.execute("SELECT COUNT(*) FROM episodes WHERE video_id = ? AND status = 'located'", (vid,)).fetchone()[0]
        if left == 0:
            state.transition(con, vid, "downloaded", "cut", note=f"{ok} clips registered, {bad} failed, {secs / 60:.0f} min")
            log(f"cut {vid} ({m['speaker']}): {ok} clips, {bad} failed, {secs / 60:.0f} min of episode audio")
        else:
            log(f"{vid}: {left} episode(s) still 'located' after cutting -- row stays 'downloaded'")
    con.close()
    log(f"done: {stats['meetings']} meetings, {stats['clips']} clips ({stats['clip_s'] / 3600:.2f} h), {stats['failed']} failed; "
        f"YouTube {stats['yt_requests']} requests {stats['yt_bytes'] / 1e6:.0f} MB {stats['yt_wall']:.0f} s; "
        f"Granicus {stats['gr_requests']} slices {stats['gr_mb']:.1f} MB" + (f"; STOPPED: {stats['bot']}" if stats["bot"] else ""))
    return EXIT_BOT_CHECK if stats["bot"] else 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max", type=int, default=0)
    ap.add_argument("--bodies", default=None)
    ap.add_argument("--min-gap", type=float, default=45)
    ap.add_argument("--max-gap", type=float, default=180)
    ap.add_argument("--plan", action="store_true", help="print the plan, no network, no cuts")
    ap.add_argument("--requeue-failed", action="store_true", help="failed episodes back to 'located' (human request), then exit")
    ap.add_argument("--state", default=None)
    return run(ap.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
