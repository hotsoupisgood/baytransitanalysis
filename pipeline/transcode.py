#!/usr/bin/env python3
"""
pipeline/transcode.py -- 'downloaded' -> 'transcoded' (PLAN.md 4.3, DECISIONS.md 4).

For each in-frame 'downloaded' row:

    ffmpeg -nostdin -t <cap> -i <audio> -ac 1 -ar 16000 -c:a libopus -b:a 16k -y output/opus/<id>.opus

`-t` applies the per-video cap (TRANSCRIBE_CAP_MIN, default 45 min; SAMPLING.md)
at transcode time, so nothing longer than the cap ever reaches the GPU. This run
takes the FIRST cap minutes (window_start_s = 0) as Puck decided on 2026-09-18;
a random window would be `-ss <offset>` here plus the window_start_s column.

The m4a is KEPT (PLAN.md 4.3): 16 kbps opus is lossless for ASR and useless for
anything acoustic, and re-downloading costs weeks. Nothing in this file deletes.

The result is verified with ffprobe (codec opus, 1 channel, duration within 1 s
of min(duration, cap)) before the row moves; transcribed_s records how much of
the video is in the file, which is what the output contract reports.

LEVEL (2026-09-23, after b17 lost 30 Fremont clips to a VAD that heard nothing).
Before encoding, the mono signal this command is about to produce is measured
with `-af volumedetect`, and two things can be inserted ahead of the encoder:

  downmix   `-ac 1` averages the channels. Fremont's Granicus MP4s are
            POLARITY-INVERTED stereo (L = -R): the average cancels the speech
            and drops the mean level by 35 dB, which is what actually killed
            b17 -- the source channels themselves read a healthy -43 dB with
            -7 dB peaks. When the summed downmix sits POLARITY_DROP_DB or more
            below channel 0 alone, one channel is taken instead of the average
            (`pan=mono|c0=c0`). Measured on b17: 32/32 Fremont clips inverted,
            51/51 clips from the other six bodies unaffected.
  loudnorm  a mono signal quieter than NORMALISE_BELOW_DBFS is brought up to
            LOUDNORM_I LUFS (EBU R128). This is the step Puck asked for; it is
            insurance for genuinely quiet rooms, NOT the b17 fix.

Both are decided per file from measurement, both are recorded, and a file above
the threshold with ordinary channels encodes byte-for-byte as it did before.

    python3 pipeline/transcode.py            # transcode everything waiting, then exit
    python3 pipeline/transcode.py --loop     # keep polling (launchd)
    python3 pipeline/transcode.py --check    # re-verify every existing opus (pre-GPU QA)
    python3 pipeline/transcode.py --levels <file>...   # what the encoder would do, $0
"""
import argparse, json, os, re, subprocess, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import state  # noqa: E402
from notify import notify  # noqa: E402

ROOT = state.ROOT

# A stereo pair whose sum is this far below channel 0 alone is out of phase; the
# average is cancellation, not a downmix. Real dual-mono measures 0.0 dB here.
POLARITY_DROP_DB = 10.0
MEASURE_TIMEOUT_S = 1800


def opus_dir():
    return Path(os.path.expanduser(state.setting("OPUS_DIR", str(ROOT / "output" / "opus"))))


def cap_seconds(kind="show"):
    """The per-video cap applies to the show frame only (SAMPLING.md). A
    kind='meeting' or 'episode' row is never capped (2026-09-21, PROPOSAL 3a):
    episodes are cut to their own length by pipeline/episodes.py, and a whole
    meeting must stay whole if it ever reaches this stage."""
    if kind in ("meeting", "episode"):
        return float("inf")
    return float(state.setting("TRANSCRIBE_CAP_MIN", 45)) * 60


def normalise_threshold_dbfs():
    """Mono mean_volume below this is normalised. Puck set -30 dB on 2026-09-23.

    Note for whoever changes it: the b17 clips that PASSED QA have a median mono
    mean_volume of -38.5 dB, so -30 dB sits ABOVE the working corpus and most
    clips will be normalised. -45 dB fires only on genuinely broken audio."""
    return float(state.setting("NORMALISE_BELOW_DBFS", -30.0))


def loudnorm_target_lufs():
    return float(state.setting("LOUDNORM_I", -16.0))


def ffmpeg_command(src, dst, cap_s, af=None):
    cap = [] if cap_s == float("inf") else ["-t", f"{cap_s:.0f}"]
    filt = ["-af", af] if af else []
    return ["ffmpeg", "-nostdin", "-v", "error", *cap, "-i", str(src), *filt,
            "-ac", "1", "-ar", "16000", "-c:a", "libopus", "-b:a", "16k", "-y", str(dst)]


# --- level measurement and the two corrections -------------------------------------

def volumedetect(src, cap_s=float("inf"), af=None, input_args=()):
    """{'mean': -43.2, 'max': -9.8} for `src` as `af` would leave it, or None.
    Decode only -- nothing is written. `input_args` are extra ffmpeg INPUT
    options (before -i), e.g. ['-user_agent', UA] for a Granicus URL, which
    403s without one, or ['-ss', '300'] to measure past an opening title card."""
    cap = [] if cap_s == float("inf") else ["-t", f"{cap_s:.0f}"]
    chain = f"{af},volumedetect" if af else "volumedetect"
    try:
        r = subprocess.run(["ffmpeg", "-nostdin", "-hide_banner", *input_args, *cap, "-i", str(src),
                            "-af", chain, "-f", "null", "-"],
                           capture_output=True, text=True, timeout=MEASURE_TIMEOUT_S)
    except (OSError, subprocess.TimeoutExpired):
        return None
    mean = re.search(r"mean_volume:\s*(-?[\d.]+) dB", r.stderr)
    peak = re.search(r"max_volume:\s*(-?[\d.]+) dB", r.stderr)
    if not mean:
        return None
    return {"mean": float(mean.group(1)), "max": float(peak.group(1)) if peak else None}


def downmix_filter(src, cap_s=float("inf"), input_args=(), probe_args=None):
    """(filter_or_None, reason). None means `-ac 1` alone is correct, which is
    the case for mono, for real dual-mono and for anything not stereo.
    `input_args` is passed to volumedetect (see there). `probe_args` is for
    ffprobe, which takes a different option set -- this build has no `-ss`, so
    a seek passed to both silently loses the channel count and the file reads
    as "not stereo"; it defaults to `input_args`."""
    p = probe(src, input_args if probe_args is None else probe_args)
    if not p or p["channels"] != 2:
        return None, "not stereo"
    one = volumedetect(src, cap_s, "pan=mono|c0=c0", input_args)
    summed = volumedetect(src, cap_s, "pan=mono|c0=0.5*c0+0.5*c1", input_args)
    if not one or not summed:
        return None, "channel probe failed"
    drop = summed["mean"] - one["mean"]
    if drop <= -POLARITY_DROP_DB:
        return "pan=mono|c0=c0", f"polarity-inverted stereo: sum {drop:.1f} dB below channel 0"
    return None, f"stereo sum {drop:.1f} dB vs channel 0"


def encode_plan(src, cap_s=float("inf")):
    """What the encoder will do to `src` and why, without writing anything.

    {'af': <filter chain or None>, 'pre_mean_dbfs': .., 'pre_max_dbfs': ..,
     'normalised': bool, 'downmix': <reason>, 'target_lufs': ..}"""
    af, why = downmix_filter(src, cap_s)
    pre = volumedetect(src, cap_s, af)
    plan = {"af": af, "downmix": why, "normalised": False,
            "pre_mean_dbfs": pre["mean"] if pre else None,
            "pre_max_dbfs": pre["max"] if pre else None,
            "threshold_dbfs": normalise_threshold_dbfs(), "target_lufs": None}
    if pre and pre["mean"] < plan["threshold_dbfs"]:
        target = loudnorm_target_lufs()
        # One-pass loudnorm: the two-pass form would double the decode for a
        # correction whose only consumer is a VAD. TP -1.5 dBTP, LRA 11 = R128.
        ln = f"loudnorm=I={target:.1f}:TP=-1.5:LRA=11"
        plan["af"] = f"{af},{ln}" if af else ln
        plan["normalised"] = True
        plan["target_lufs"] = target
    return plan


def encode(src, dst, cap_s, measure_after=True):
    """The clip -> opus step for every caller. Returns (ok, detail, levels);
    `levels` is encode_plan's dict plus post_mean_dbfs / post_max_dbfs, and is
    what the episode row records. Writes only `dst`; `src` is never touched."""
    plan = encode_plan(src, cap_s)
    r = subprocess.run(ffmpeg_command(src, dst, cap_s, plan["af"]),
                       capture_output=True, text=True, timeout=MEASURE_TIMEOUT_S)
    if r.returncode != 0:
        return False, (r.stderr or "").strip()[-200:] or f"ffmpeg exit {r.returncode}", plan
    if measure_after:
        post = volumedetect(dst)
        plan["post_mean_dbfs"] = post["mean"] if post else None
        plan["post_max_dbfs"] = post["max"] if post else None
    return True, level_note(plan), plan


def level_note(p):
    """The one-line human form that goes in the events note."""
    bits = []
    if p.get("pre_mean_dbfs") is not None:
        bits.append(f"pre {p['pre_mean_dbfs']:.1f} dB")
    if p.get("post_mean_dbfs") is not None:
        bits.append(f"post {p['post_mean_dbfs']:.1f} dB")
    if p.get("af") and "pan=" in (p["af"] or ""):
        bits.append("single-channel (inverted stereo)")
    bits.append(f"normalised to {p['target_lufs']:.0f} LUFS" if p.get("normalised") else "level unchanged")
    return ", ".join(bits)


def probe(path, input_args=()):
    """{'codec': 'opus', 'channels': 1, 'duration': 2700.0} or None. Opus always
    REPORTS 48 kHz; the 16 kHz input is what matters (PLAN.md 4.3). `channels`
    is the LAST stream's, i.e. the audio one in a video+audio MP4; `input_args`
    are extra ffprobe input options (see volumedetect)."""
    try:
        r = subprocess.run(["ffprobe", "-v", "error", *input_args, "-show_entries", "format=duration:stream=codec_name,channels",
                            "-of", "default=nw=1", str(path)], capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if r.returncode != 0:
        return None
    kv = dict(l.split("=", 1) for l in r.stdout.splitlines() if "=" in l)
    try:
        return {"codec": kv.get("codec_name"), "channels": int(kv.get("channels", 0)),
                "duration": float(kv.get("duration", 0))}
    except ValueError:
        return None


def verify(path, expected_s, tol=1.5):
    """Returns (ok, detail)."""
    p = probe(path)
    if not p:
        return False, "ffprobe cannot read the opus file"
    if p["codec"] != "opus" or p["channels"] != 1:
        return False, f"codec={p['codec']} channels={p['channels']} (want opus, 1)"
    if abs(p["duration"] - expected_s) > tol:
        return False, f"duration {p['duration']:.1f} s, expected {expected_s:.1f} s"
    return True, f"{p['duration']:.1f} s, {Path(path).stat().st_size / 1e6:.2f} MB"


def transcode_one(con, video_id, cap_s, out_dir):
    row = con.execute("SELECT audio_path, duration_s, kind FROM videos WHERE video_id = ?", (video_id,)).fetchone()
    src = Path(row["audio_path"] or "")
    if not src.exists():
        return False, f"audio file missing: {src}"
    dur = float(row["duration_s"] or 0) or (probe(src) or {}).get("duration") or 0
    if row["kind"] in ("meeting", "episode"):
        cap_s = cap_seconds(row["kind"])
    expected = min(dur, cap_s)
    dst = out_dir / f"{video_id}.opus"
    tmp = out_dir / f"{video_id}.tmp.opus"        # ffmpeg picks the container from the suffix
    t0 = time.time()
    ok, detail, levels = encode(src, tmp, cap_s)
    if not ok:
        tmp.unlink(missing_ok=True)
        return False, f"ffmpeg: {detail}"
    level_detail = detail
    ok, detail = verify(tmp, expected)
    if not ok:
        tmp.unlink(missing_ok=True)
        return False, detail
    tmp.replace(dst)
    return True, (f"{detail}, {level_detail}", str(dst), expected, time.time() - t0, levels)


def log(msg):
    print(f"{state.now_iso()} {msg}", flush=True)


def run(args):
    con = state.connect(args.state)
    out_dir = opus_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    cap_s = cap_seconds()
    done = 0
    while True:
        row = con.execute("SELECT video_id FROM videos WHERE in_frame = 1 AND status = 'downloaded' "
                          "ORDER BY updated_at LIMIT 1").fetchone()
        if not row:
            if not args.loop:
                break
            time.sleep(args.idle_sleep)
            continue
        vid = row["video_id"]
        # No intermediate state: transcode is seconds long and idempotent, so a
        # crash mid-file leaves the row 'downloaded' and a .tmp.opus we overwrite.
        ok, res = transcode_one(con, vid, cap_s, out_dir)
        if ok:
            detail, dst, transcribed_s, secs, _levels = res
            state.transition(con, vid, "downloaded", "transcoded", note=f"{detail}, {secs:.0f} s",
                             opus_path=dst, transcribed_s=transcribed_s)
            done += 1
            log(f"transcoded {vid}: {detail} ({secs:.0f} s)")
        else:
            new = state.fail(con, vid, "downloaded", res)
            log(f"{new.upper()} {vid}: {res}")
            if new == "dead":
                notify("error", f"{vid} is dead after {state.MAX_ATTEMPTS} transcode attempts: {res}")
        if args.max and done >= args.max:
            break
    con.close()
    log(f"done: {done} transcoded")
    return 0


def check(args):
    """Pre-GPU acceptance table (docs/TESTING.md 'Output acceptance'): every
    transcoded/queued row's opus re-probed, MB per audio-hour shown."""
    con = state.connect(args.state)
    bad = 0
    print(f"{'video_id':<12} {'speaker':<9} {'dur_min':>7} {'opus_s':>7} {'MB':>6} {'MB/h':>5}  result")
    for r in con.execute("SELECT video_id, speaker, duration_s, transcribed_s, opus_path FROM videos "
                         "WHERE in_frame = 1 AND status IN ('transcoded', 'queued', 'transcribing', 'done') "
                         "ORDER BY speaker, video_id"):
        ok, detail = verify(r["opus_path"] or "", float(r["transcribed_s"] or 0))
        mb = Path(r["opus_path"]).stat().st_size / 1e6 if r["opus_path"] and Path(r["opus_path"]).exists() else 0
        hrs = float(r["transcribed_s"] or 0) / 3600
        bad += 0 if ok else 1
        print(f"{r['video_id']:<12} {r['speaker']:<9} {float(r['duration_s'] or 0) / 60:7.1f} "
              f"{float(r['transcribed_s'] or 0):7.0f} {mb:6.2f} {mb / hrs if hrs else 0:5.1f}  {'OK' if ok else 'FAIL'} {detail}")
    print(f"{bad} problem(s)")
    return 1 if bad else 0


def levels(args):
    """What the encoder would do to each file, measured, without writing anything."""
    print(f"{'file':<34}{'mean dB':>9}{'max dB':>8}  action")
    for f in args.levels:
        p = encode_plan(f)
        act = []
        if p["af"] and "pan=" in p["af"]:
            act.append(p["downmix"])
        if p["normalised"]:
            act.append(f"-> loudnorm I={p['target_lufs']:.0f}")
        mean = "-" if p["pre_mean_dbfs"] is None else f"{p['pre_mean_dbfs']:.1f}"
        peak = "-" if p["pre_max_dbfs"] is None else f"{p['pre_max_dbfs']:.1f}"
        print(f"{Path(f).name:<34}{mean:>9}{peak:>8}  {'; '.join(act) or 'encode unchanged'}")
        if args.json:
            print(json.dumps(p))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--loop", action="store_true")
    ap.add_argument("--max", type=int, default=0)
    ap.add_argument("--idle-sleep", type=float, default=60)
    ap.add_argument("--check", action="store_true", help="re-verify existing opus files and print the table")
    ap.add_argument("--levels", nargs="+", metavar="FILE", help="measure files and print the encode plan; writes nothing")
    ap.add_argument("--json", action="store_true", help="with --levels, also print the plan as JSON")
    ap.add_argument("--state", default=None)
    args = ap.parse_args(argv)
    if args.levels:
        return levels(args)
    return check(args) if args.check else run(args)


if __name__ == "__main__":
    sys.exit(main())
