#!/usr/bin/env python3
"""
scripts/t0_fetch_episodes.py -- T0 test 4, second runner (plan/T0-TESTPLAN.md
test 4; supersedes output/t0/download_sections.py after its 403).

WHY: the first runner asked YouTube for one `--download-sections` range per
episode. The 6th range request on the same video (9FM5406YagE) got HTTP 403
from googlevideo -- not a bot-check, a per-video limit on repeated stream
opens. So: for a meeting with >= WHOLE_MIN planned episodes, fetch the WHOLE
audio once (`-f bestaudio -x --audio-format opus`, one player request) and cut
the episodes locally with `ffmpeg -ss/-t -c copy` (Ogg Opus copy-cuts are
packet-accurate, ~20 ms; re-encode only if the copy misaligns by > 1 s).
Meetings with 1-2 episodes still use `--download-sections`. Clips that already
exist are kept and skipped.

Layout: output/t0/<video_id>/full.opus (whole meeting, kept -- input for a
test-5 batch), output/t0/<video_id>/e<n>.opus (episode with 30 s pads),
output/t0/<video_id>/full.log | e<n>.log (yt-dlp / ffmpeg output).

YouTube rules (AGENTS.md, T0-TESTPLAN.md): this Mac only, strictly sequential,
`--sleep-requests 2`, random 45-180 s gap between fetches, yt-dlp's own
retries capped at 2. Stop on "Sign in to confirm", HTTP 429, or a SECOND 403;
a first 403 is logged and the run moves to the next video. Never loop retries.

    nohup .venv/bin/python scripts/t0_fetch_episodes.py > output/t0/fetch_episodes.log 2>&1 &
    .venv/bin/python scripts/t0_fetch_episodes.py --plan           # what would be fetched, no network
    .venv/bin/python scripts/t0_fetch_episodes.py --report-only    # ffprobe + edge check -> fetch_report.json

State: output/t0/fetch_rows.json (one row per clip), fetch_requests.json (one
row per YouTube request: kind, wall, bytes, rc, block signal). Both are
rewritten after every step, so a killed run resumes.
"""
import argparse, json, random, re, subprocess, sys, time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pipeline"))
import segment as sg                                             # noqa: E402

T0 = ROOT / "output" / "t0"
Y = str(ROOT / ".venv" / "bin" / "yt-dlp")
PAD = 30.0
CAP = 40
WHOLE_MIN = 3            # >= this many planned episodes on one video -> whole-audio strategy
COPY_TOL_S = 1.0         # copy-cut duration error above this -> re-encode
BLOCK = re.compile(r"sign in to confirm|HTTP Error 429|Too Many Requests", re.I)
FORBIDDEN = re.compile(r"HTTP Error 403|403 Forbidden", re.I)
RATE = re.compile(r"\[download\] 100% of\s+([\d.]+)(\w+) in ([\d:]+) at\s+([\d.]+)(\w+)/s")
ROWS, REQS, REPORT, OLD_ROWS = T0 / "fetch_rows.json", T0 / "fetch_requests.json", T0 / "fetch_report.json", T0 / "sections_rows.json"


def now():
    return datetime.now().strftime("%Y-%m-%dT%H:%M:%S")


def plan():
    """The CAP longest about_* episodes, grouped by video, whole-audio videos
    first (biggest transfers early), 9FM5406YagE last (it had the 403)."""
    eps = json.loads((T0 / "episodes.json").read_text())
    eps = sorted(eps, key=lambda e: -e["duration_s"])[:CAP]
    by_vid = {}
    for e in sorted(eps, key=lambda e: (e["video_id"], e["start_s"])):
        by_vid.setdefault(e["video_id"], []).append(e)
    order = sorted(by_vid, key=lambda v: (v == "9FM5406YagE", len(by_vid[v]) < WHOLE_MIN, v))
    return [(v, "whole" if len(by_vid[v]) >= WHOLE_MIN else "section", by_vid[v]) for v in order]


def clip_path(e):
    n = e["episode_id"].rsplit("_e", 1)[1]
    return T0 / e["video_id"] / f"e{n}.opus"


def padded(e):
    return max(0.0, e["start_s"] - PAD), e["end_s"] + PAD


def run_ytdlp(args, log_path, timeout):
    t = time.time()
    try:
        p = subprocess.run([Y, "--sleep-requests", "2", "--no-playlist", "--retries", "2", "--fragment-retries", "2", *args],
                           capture_output=True, text=True, timeout=timeout)
        rc, log = p.returncode, p.stdout + p.stderr
    except subprocess.TimeoutExpired as ex:
        rc, log = -9, (ex.stdout or "") + (ex.stderr or "") + f"\n!!! runner timeout after {timeout} s\n"
    log_path.write_text(log)
    m = RATE.search(log)
    return {"rc": rc, "wall_s": round(time.time() - t, 1), "blocked": bool(BLOCK.search(log)), "forbidden": bool(FORBIDDEN.search(log)),
            "ytdlp_rate": f"{m.group(4)} {m.group(5)}/s" if m else None, "ytdlp_transfer_time": m.group(3) if m else None}


def fetch_whole(vid):
    out = T0 / vid / "full"
    r = run_ytdlp(["-f", "bestaudio", "-x", "--audio-format", "opus", "-o", str(out) + ".%(ext)s", f"https://www.youtube.com/watch?v={vid}"],
                  T0 / vid / "full.log", timeout=6 * 3600)
    p = Path(str(out) + ".opus")
    return {"kind": "whole", "video_id": vid, "at": now(), "path": str(p), "exists": p.exists(), "bytes": p.stat().st_size if p.exists() else 0, **r}


def fetch_section(e):
    vid, a, b = e["video_id"], *padded(e)
    p = clip_path(e)
    r = run_ytdlp(["--download-sections", f"*{a}-{b}", "-f", "bestaudio", "-x", "--audio-format", "opus",
                   "-o", str(p.with_suffix("")) + ".%(ext)s", f"https://www.youtube.com/watch?v={vid}"], p.with_suffix(".log"), timeout=1800)
    return {"kind": "section", "video_id": vid, "episode_id": e["episode_id"], "at": now(), "path": str(p), "exists": p.exists(),
            "bytes": p.stat().st_size if p.exists() else 0, **r}


def cut(e, full):
    """ffmpeg -ss/-t -c copy from full.opus; re-encode if the copy misaligns."""
    a, b = padded(e)
    p = clip_path(e)
    how, t = "copy", time.time()
    for codec in (["-c", "copy"], ["-c:a", "libopus", "-b:a", "112k"]):
        r = subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", str(a), "-i", str(full), "-t", str(b - a), *codec, str(p)], capture_output=True, text=True)
        d = ffprobe(str(p))["duration_s"] if p.exists() else None
        p.with_suffix(".log").write_text(f"ffmpeg -ss {a} -t {b - a} {' '.join(codec)} rc={r.returncode} duration={d}\n{r.stderr}")
        if r.returncode == 0 and d is not None and abs(d - (b - a)) <= COPY_TOL_S:
            break
        how = "reencode"
    return {"source": f"ffmpeg {how} from full.opus", "rc": r.returncode, "wall_s": round(time.time() - t, 1)}


def ffprobe(path):
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration,size,bit_rate", "-of", "json", path], capture_output=True, text=True)
    try:
        f = json.loads(r.stdout)["format"]
        return {"duration_s": round(float(f["duration"]), 2), "bytes": int(f["size"]), "bit_rate": int(f.get("bit_rate") or 0)}
    except Exception:
        return {"duration_s": None, "bytes": None, "bit_rate": None}


def edge_check(vid, a, b):
    """Does a caption cue straddle the clip boundary? (auto captions have no
    punctuation, so a straddling cue is the mid-sentence proxy). Also the gap
    to the nearest cue edge and whether a '>>' speaker change is within 5 s."""
    vtt = next((T0 / vid).glob("*.en.vtt"), None) or next((T0 / vid).glob("*.vtt"), None)
    segs = sg.read_segments(vtt) if vtt else []

    def at(t):
        strad = [s for s in segs if s["start"] < t < s["end"]]
        near = min((abs(s["start"] - t) for s in segs), default=None)
        spk = any(">>" in s["text"] and abs(s["start"] - t) <= 5 for s in segs)
        return {"cue_straddles": bool(strad), "nearest_cue_edge_s": round(near, 2) if near is not None else None, "speaker_change_within_5s": spk,
                "text": " ".join(s["text"] for s in segs if t - 8 <= s["start"] <= t + 8)[:160]}
    return {"start": at(a), "end": at(b)}


def load(path, default):
    return json.loads(path.read_text()) if path.exists() else default


def row_for(e, strategy, source, rc, wall_s):
    a, b = padded(e)
    p = clip_path(e)
    return {"episode_id": e["episode_id"], "video_id": e["video_id"], "start_s": a, "end_s": b, "core_start_s": e["start_s"], "core_end_s": e["end_s"],
            "strategy": strategy, "source": source, "rc": rc, "wall_s": wall_s, "path": str(p), "exists": p.exists()}


def upsert(rows, r):
    return [x for x in rows if x["episode_id"] != r["episode_id"]] + [r]


def report(rows, reqs, planned):
    for r in rows:
        if r.get("exists"):
            r.update(ffprobe(r["path"]))
            # --download-sections clips start early (ffmpeg seeks to the webm cluster before -ss: measured
            # 10.0 s on every one, cross-correlated against full.opus) and end on time; copy-cuts are exact.
            r["actual_start_s"] = round(r["end_s"] - r["duration_s"], 2) if r["strategy"] == "section" else r["start_s"]
            r["edges"] = edge_check(r["video_id"], r["actual_start_s"], r["end_s"])
    ok = [r for r in rows if r.get("exists")]
    tot = {"planned": planned, "fetched": len(ok), "missing": planned - len(ok),
           "audio_hours": round(sum(r["duration_s"] or 0 for r in ok) / 3600, 3), "MB": round(sum(r["bytes"] or 0 for r in ok) / 1e6, 2),
           "start_cut_mid_cue": sum(r["edges"]["start"]["cue_straddles"] for r in ok), "end_cut_mid_cue": sum(r["edges"]["end"]["cue_straddles"] for r in ok),
           "start_near_speaker_change": sum(r["edges"]["start"]["speaker_change_within_5s"] for r in ok),
           "end_near_speaker_change": sum(r["edges"]["end"]["speaker_change_within_5s"] for r in ok),
           "by_strategy": {s: len([r for r in ok if r["strategy"] == s]) for s in ("whole", "section")}}
    if ok:
        tot["MB_per_audio_hour"] = round(tot["MB"] / tot["audio_hours"], 1) if tot["audio_hours"] else None
        tot["mean_kbps"] = round(sum(r["bit_rate"] or 0 for r in ok) / len(ok) / 1000, 1)
    # YouTube requests: whole vs section
    by = {}
    for q in reqs:
        d = by.setdefault(q["kind"], {"requests": 0, "ok": 0, "bytes": 0, "wall_s": 0.0, "http403": 0, "blocked": 0})
        d["requests"] += 1; d["ok"] += bool(q.get("exists")); d["bytes"] += q.get("bytes") or 0; d["wall_s"] += q.get("wall_s") or 0
        d["http403"] += bool(q.get("forbidden")); d["blocked"] += bool(q.get("blocked"))
    for d in by.values():
        d["MB"] = round(d["bytes"] / 1e6, 1); d["MB_per_request"] = round(d["MB"] / d["requests"], 1) if d["requests"] else None
        d["KB_per_s"] = round(d["bytes"] / 1024 / d["wall_s"], 1) if d["wall_s"] else None
        d["wall_s"] = round(d["wall_s"], 1)
    fulls = {q["video_id"]: q for q in reqs if q["kind"] == "whole" and q.get("exists")}
    for vid, q in fulls.items():
        q["full_duration_s"] = ffprobe(q["path"])["duration_s"]
        used = sum(r["end_s"] - r["start_s"] for r in ok if r["video_id"] == vid and r["strategy"] == "whole")
        q["episode_s_cut"] = round(used, 1); q["share_used"] = round(used / q["full_duration_s"], 3) if q["full_duration_s"] else None
    tot["requests"] = by
    tot["whole_audio"] = {"videos": len(fulls), "hours": round(sum(q["full_duration_s"] or 0 for q in fulls.values()) / 3600, 2),
                          "MB": round(sum(q["bytes"] for q in fulls.values()) / 1e6, 1),
                          "episode_hours_cut": round(sum(q["episode_s_cut"] for q in fulls.values()) / 3600, 2)}
    tot["wall_s_requests"] = round(sum(q.get("wall_s") or 0 for q in reqs), 1)
    REPORT.write_text(json.dumps({"total": tot, "clips": sorted(rows, key=lambda r: (r["video_id"], r["start_s"])), "requests": reqs}, indent=1))
    print(json.dumps(tot, indent=1))
    return tot


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", action="store_true", help="print the plan, no network")
    ap.add_argument("--report-only", action="store_true")
    a = ap.parse_args(argv)
    groups = plan()
    planned = sum(len(es) for _, _, es in groups)
    rows, reqs = load(ROWS, []), load(REQS, [])
    if not rows and OLD_ROWS.exists():                       # clips from the first runner (2026-09-20) are kept as-is
        for r in load(OLD_ROWS, []):
            if r.get("exists") and Path(r["path"]).exists():
                rows.append({**{k: r[k] for k in ("episode_id", "video_id", "start_s", "end_s", "core_start_s", "core_end_s", "rc", "wall_s", "path", "exists")},
                             "strategy": "section", "source": "yt-dlp --download-sections (runner 1, 2026-09-20)"})
                reqs.append({"kind": "section", "video_id": r["video_id"], "episode_id": r["episode_id"], "at": "2026-09-20 runner 1", "path": r["path"],
                             "exists": True, "bytes": Path(r["path"]).stat().st_size, "rc": r["rc"], "wall_s": r["wall_s"], "blocked": False, "forbidden": False})
            elif r.get("blocked"):
                reqs.append({"kind": "section", "video_id": r["video_id"], "episode_id": r["episode_id"], "at": "2026-09-20 runner 1", "path": r["path"],
                             "exists": False, "bytes": 0, "rc": r["rc"], "wall_s": r["wall_s"], "blocked": False, "forbidden": True})
    if a.report_only:
        report(rows, reqs, planned); return 0
    have = {r["episode_id"] for r in rows if r.get("exists")}
    print(f"{now()} {planned} episodes planned (cap {CAP}), {len(have)} already on disk, pad {PAD} s, whole-audio when >= {WHOLE_MIN} episodes", flush=True)
    for vid, strategy, es in groups:
        todo = [e for e in es if e["episode_id"] not in have]
        full = T0 / vid / "full.opus"
        print(f"  {vid} {strategy:7s} {len(es)} episodes, {len(todo)} to fetch, {round(sum(e['end_s'] - e['start_s'] + 2 * PAD for e in todo) / 60)} min"
              + (" (full.opus present)" if full.exists() else ""), flush=True)
    if a.plan:
        return 0
    t_all, n403, fetched_any = time.time(), sum(bool(q.get("forbidden")) for q in reqs if q.get("at") != "2026-09-20 runner 1"), False

    def gap():
        g = random.randint(45, 180)
        print(f"   sleep {g}", flush=True); time.sleep(g)

    def log_req(q):
        print(f"REQ {q['at']} {q['kind']} {q['video_id']} {q.get('episode_id', '')} rc={q['rc']} wall={q['wall_s']} s bytes={q['bytes']} "
              f"rate={q['ytdlp_rate']} 403={q['forbidden']} block={q['blocked']}", flush=True)

    stop = None
    for vid, strategy, es in groups:
        todo = [e for e in es if e["episode_id"] not in have]
        if not todo:
            continue
        full = T0 / vid / "full.opus"
        if strategy == "whole":
            if not full.exists():
                if fetched_any: gap()
                q = fetch_whole(vid); fetched_any = True
                reqs.append(q); REQS.write_text(json.dumps(reqs, indent=1)); log_req(q)
                if q["blocked"] or q["rc"] == -9:
                    stop = "bot-check / 429 / timeout"; break
                if q["forbidden"]:
                    n403 += 1
                    if n403 >= 2: stop = "second HTTP 403"; break
                    print(f"   first 403 on {vid} -- skipping this video, continuing", flush=True); continue
                if not q["exists"]:
                    print(f"   {vid}: yt-dlp rc={q['rc']} and no full.opus -- skipping this video", flush=True); continue
            for e in todo:
                c = cut(e, full)
                r = row_for(e, "whole", c["source"], c["rc"], c["wall_s"]); rows = upsert(rows, r); ROWS.write_text(json.dumps(rows, indent=1))
                d = ffprobe(r["path"])["duration_s"] if r["exists"] else None
                print(f"CUT {r['episode_id']} {sg.hms(r['start_s'])}-{sg.hms(r['end_s'])} {c['source']} rc={c['rc']} {c['wall_s']} s dur={d}", flush=True)
        else:
            for e in todo:
                if fetched_any: gap()
                q = fetch_section(e); fetched_any = True
                reqs.append(q); REQS.write_text(json.dumps(reqs, indent=1)); log_req(q)
                r = row_for(e, "section", "yt-dlp --download-sections", q["rc"], q["wall_s"]); rows = upsert(rows, r); ROWS.write_text(json.dumps(rows, indent=1))
                if q["blocked"] or q["rc"] == -9:
                    stop = "bot-check / 429 / timeout"; break
                if q["forbidden"]:
                    n403 += 1
                    if n403 >= 2: stop = "second HTTP 403"; break
                    print(f"   first 403 on {vid} -- moving on", flush=True); break
            if stop: break
    if stop:
        print(f"!!! STOPPED ({stop}) at {now()} -- not retrying. See the last log under output/t0/<video_id>/.", flush=True)
    print(f"{now()} wall total {round(time.time() - t_all)} s", flush=True)
    report(rows, reqs, planned)
    return 3 if stop else 0


if __name__ == "__main__":
    sys.exit(main())
