#!/usr/bin/env python3
"""
scrape_hasan_audio.py
======================
Bulk-download every video's audio from a YouTube channel (default: full
HasanAbi back-catalog) straight to an external drive. No transcription, no
diffing — just audio, resumable across runs.

Quick start:
    python3 legacy/scrape_hasan_audio.py --list-only          # see how much there is first
    python3 legacy/scrape_hasan_audio.py                       # start/resume the real pull

Safe to interrupt (Ctrl-C) and re-run any time: a download archive file next
to the audio tracks which video IDs are already done, so re-running only
fetches what's missing. Stops gracefully (not mid-write) if free space on the
destination drive drops below --min-free-gb.

Output (on the destination drive, default /Volumes/Extreme SSD/hasan_audio):
    <upload_date>_<id>.m4a         one file per video
    downloaded.archive.txt         yt-dlp-style archive of completed IDs
    failed.txt                     IDs that errored this run (retried next run)
    scrape.log                     append-only run log
    meta.json                      channel + sampling settings for this folder

Each destination folder is self-contained and portable: move it anywhere
(a different drive, a different machine) and re-run with just --dest pointing
at its new location — meta.json supplies the original --channel/--every-nth
so it resumes correctly without those flags needing to be remembered/passed
again, and downloaded.archive.txt means nothing already-downloaded repeats.
"""

import argparse
import json
import random
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

DEFAULT_CHANNEL = "https://www.youtube.com/@hasanabi/videos"
DEFAULT_DEST = "/Volumes/Extreme SSD/hasan_audio"


def log(msg: str, log_path: Path | None = None):
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(line)
    if log_path:
        with log_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")


def list_all_videos(channel: str):
    """Enumerate every video on the channel tab via yt-dlp's flat playlist mode.
    Metadata-only — no media is pulled here."""
    cmd = [
        "yt-dlp", "--flat-playlist", "--ignore-errors",
        "--print", "%(id)s\t%(title)s\t%(duration)s\t%(upload_date)s",
        channel,
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    out = []
    for line in res.stdout.strip().splitlines():
        parts = line.split("\t", 3)
        if len(parts) != 4:
            continue
        vid, title, dur, upload_date = parts
        try:
            dur = int(float(dur)) if dur and dur != "NA" else 0
        except ValueError:
            dur = 0
        out.append({"id": vid, "title": title, "duration": dur, "upload_date": upload_date or "00000000"})
    return out


def load_id_set(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {l.strip() for l in path.read_text(encoding="utf-8").splitlines() if l.strip()}


def append_id(path: Path, vid: str):
    with path.open("a", encoding="utf-8") as f:
        f.write(vid + "\n")


def free_gb(path: Path) -> float:
    return shutil.disk_usage(path).free / 1e9


def download_one(vid: dict, dest: Path, cookies_from_browser: str | None, sleep_requests: float):
    url = f"https://www.youtube.com/watch?v={vid['id']}"
    out_tmpl = str(dest / "%(upload_date)s_%(id)s.%(ext)s")
    cmd = [
        "yt-dlp",
        "-f", "bestaudio[ext=m4a]/bestaudio",
        "--no-playlist",
        "--sleep-requests", str(sleep_requests),
        "-o", out_tmpl,
        url,
    ]
    if cookies_from_browser:
        cmd[1:1] = ["--cookies-from-browser", cookies_from_browser]
    return subprocess.run(cmd, capture_output=True, text=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--channel", default=None,
                    help=f"defaults to meta.json's channel if the dest folder has one, else {DEFAULT_CHANNEL}")
    ap.add_argument("--dest", default=DEFAULT_DEST)
    ap.add_argument("--min-free-gb", type=float, default=20.0,
                    help="stop gracefully once free space on dest drive drops below this")
    ap.add_argument("--sleep-min", type=float, default=4.0, help="min seconds between downloads")
    ap.add_argument("--sleep-max", type=float, default=10.0, help="max seconds between downloads")
    ap.add_argument("--list-only", action="store_true",
                    help="just enumerate the channel and print total count/duration, no downloads")
    ap.add_argument("--cookies-from-browser", default=None,
                    help="pass-through to yt-dlp, e.g. 'chrome' or 'safari', if YouTube starts bot-checking you")
    ap.add_argument("--sleep-requests", type=float, default=1.5,
                    help="yt-dlp --sleep-requests: pause between internal webpage/API calls within one video's extraction")
    ap.add_argument("--every-nth", type=int, default=None,
                    help="only keep every Nth video from the channel listing (systematic sample), e.g. 10 for a 1-in-10 sample. Defaults to meta.json's value if the dest folder has one, else 1")
    args = ap.parse_args()

    dest = Path(args.dest)
    if not dest.parent.exists():
        print(f"!! {dest.parent} doesn't exist — is the drive mounted?")
        sys.exit(1)
    dest.mkdir(parents=True, exist_ok=True)

    archive_path = dest / "downloaded.archive.txt"
    failed_path = dest / "failed.txt"
    log_path = dest / "scrape.log"
    meta_path = dest / "meta.json"

    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    channel = args.channel or meta.get("channel") or DEFAULT_CHANNEL
    every_nth = args.every_nth if args.every_nth is not None else meta.get("every_nth", 1)
    if meta and (channel != meta.get("channel") or every_nth != meta.get("every_nth", 1)):
        log(f"!! overriding this folder's saved settings (channel={meta.get('channel')!r}, "
            f"every_nth={meta.get('every_nth', 1)}) with channel={channel!r}, every_nth={every_nth} — "
            f"make sure that's intentional", log_path)
    meta_path.write_text(json.dumps({
        "channel": channel,
        "every_nth": every_nth,
        "first_started": meta.get("first_started", datetime.now().isoformat()),
        "last_run": datetime.now().isoformat(),
    }, indent=2), encoding="utf-8")

    log(f"listing videos from {channel} ...", log_path)
    videos = list_all_videos(channel)
    total_hours = sum(v["duration"] for v in videos) / 3600
    log(f"found {len(videos)} videos, ~{total_hours:.1f}h total", log_path)

    if every_nth > 1:
        videos = videos[::every_nth]
        sampled_hours = sum(v["duration"] for v in videos) / 3600
        log(f"sampling every {every_nth}th video -> {len(videos)} videos, ~{sampled_hours:.1f}h", log_path)

    if args.list_only:
        return

    done = load_id_set(archive_path)
    todo = [v for v in videos if v["id"] not in done]
    log(f"{len(done)} already downloaded, {len(todo)} remaining", log_path)

    for i, v in enumerate(todo, 1):
        free = free_gb(dest)
        if free < args.min_free_gb:
            log(f"free space {free:.1f}GB < --min-free-gb {args.min_free_gb}GB — stopping. "
                f"{len(todo) - i + 1} videos left undone.", log_path)
            break

        label = f"[{i}/{len(todo)}] {v['id']} ({v['duration']//60}min) {v['title'][:60]}"
        res = download_one(v, dest, args.cookies_from_browser, args.sleep_requests)
        if res.returncode == 0:
            append_id(archive_path, v["id"])
            log(f"{label} -> ok  (free: {free:.1f}GB)", log_path)
        else:
            err = res.stderr.strip().splitlines()[-1] if res.stderr.strip() else "unknown error"
            with failed_path.open("a", encoding="utf-8") as f:
                f.write(f"{v['id']}\t{err}\n")
            log(f"{label} -> FAILED: {err}", log_path)

        time.sleep(random.uniform(args.sleep_min, args.sleep_max))

    log("done.", log_path)


if __name__ == "__main__":
    main()
